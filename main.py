import os
import hmac
import json
import base64
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from starlette.responses import FileResponse

from database import db, create_document
from schemas import (
    User as UserSchema,
    Product as ProductSchema,
    ProductVariant,
    ProductZone,
    Cart as CartSchema,
    Design as DesignSchema,
    Promo as PromoSchema,
)

from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import letter

SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-change-me").encode()
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

app = FastAPI(title="Ecommerce + Customizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ASSETS_DIR = os.path.join(os.getcwd(), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ---- Minimal JWT (HS256) ----

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    padding = '=' * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def jwt_encode(payload: dict, secret: bytes) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(',', ':')).encode())
    payload_b64 = _b64url_encode(json.dumps(payload, default=str, separators=(',', ':')).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(secret, signing_input, digestmod="sha256").digest()
    sig_b64 = _b64url_encode(signature)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def jwt_decode(token: str, secret: bytes) -> dict:
    try:
        header_b64, payload_b64, sig_b64 = token.split('.')
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token format")
    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected_sig = hmac.new(secret, signing_input, digestmod="sha256").digest()
    if not hmac.compare_digest(expected_sig, _b64url_decode(sig_b64)):
        raise HTTPException(status_code=401, detail="Invalid token signature")
    payload = json.loads(_b64url_decode(payload_b64).decode())
    if "exp" in payload:
        exp = datetime.fromisoformat(payload["exp"]).replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > exp:
            raise HTTPException(status_code=401, detail="Token expired")
    return payload


# ---- Utilities ----

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire.isoformat()})
    return jwt_encode(to_encode, SECRET_KEY)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    payload = jwt_decode(token, SECRET_KEY)
    email = payload.get("email")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db["user"].find_one({"email": email})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ---- Models for requests ----
class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AddCartItemRequest(BaseModel):
    product_id: str
    variant_sku: str
    quantity: int = 1
    price: float
    design_id: Optional[str] = None


class UpdateOrderStatusRequest(BaseModel):
    status: str


# ---- Auth ----
@app.post("/auth/signup", response_model=TokenResponse)
def signup(body: SignupRequest):
    existing = db["user"].find_one({"email": body.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_doc = UserSchema(
        email=body.email,
        password_hash=get_password_hash(body.password),
        name=body.name,
        roles=["user"],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    user_id = create_document("user", user_doc)
    token = create_access_token({"sub": user_id, "email": body.email, "roles": ["user"]})
    return TokenResponse(access_token=token)


@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest):
    user = db["user"].find_one({"email": body.email})
    if not user or not verify_password(body.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": str(user.get("_id")), "email": user["email"], "roles": user.get("roles", [])})
    return TokenResponse(access_token=token)


@app.get("/auth/me")
def me(user=Depends(get_current_user)):
    user.pop("password_hash", None)
    user["_id"] = str(user["_id"]) if "_id" in user else None
    return user


# ---- Products ----
@app.post("/admin/products")
def create_product(product: ProductSchema, user=Depends(get_current_user)):
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin only")
    doc = product.model_dump()
    doc["created_at"] = datetime.now(timezone.utc)
    doc["updated_at"] = datetime.now(timezone.utc)
    pid = create_document("product", doc)
    return {"id": pid}


@app.put("/admin/products/{product_id}")
def update_product(product_id: str, product: ProductSchema, user=Depends(get_current_user)):
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin only")
    from bson import ObjectId
    try:
        oid = ObjectId(product_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Product not found")
    doc = product.model_dump()
    doc["updated_at"] = datetime.now(timezone.utc)
    res = db["product"].update_one({"_id": oid}, {"$set": doc})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"ok": True}


@app.delete("/admin/products/{product_id}")
def delete_product(product_id: str, user=Depends(get_current_user)):
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin only")
    from bson import ObjectId
    try:
        oid = ObjectId(product_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Product not found")
    db["product"].delete_one({"_id": oid})
    return {"ok": True}


@app.get("/products")
def list_products(q: Optional[str] = None, category: Optional[str] = None, page: int = 1, per_page: int = 20, customizable: Optional[bool] = None):
    filt: Dict[str, Any] = {}
    if category:
        filt["category"] = category
    if customizable is not None:
        filt["customizable"] = customizable
    if q:
        filt["title"] = {"$regex": q, "$options": "i"}
    cursor = db["product"].find(filt).skip((page - 1) * per_page).limit(per_page)
    items = []
    for doc in cursor:
        doc["_id"] = str(doc.get("_id"))
        items.append(doc)
    total = db["product"].count_documents(filt)
    return {"items": items, "page": page, "per_page": per_page, "total": total}


@app.get("/products/{product_id}")
def get_product(product_id: str):
    from bson import ObjectId
    try:
        doc = db["product"].find_one({"_id": ObjectId(product_id)})
    except Exception:
        raise HTTPException(status_code=404, detail="Product not found")
    if not doc:
        raise HTTPException(status_code=404, detail="Product not found")
    doc["_id"] = str(doc.get("_id"))
    return doc


# ---- Cart ----
@app.get("/cart")
def get_cart(user=Depends(get_current_user)):
    cart = db["cart"].find_one({"user_id": str(user["_id"])})
    if not cart:
        cart = CartSchema(user_id=str(user["_id"]))
        create_document("cart", cart)
        cart = db["cart"].find_one({"user_id": str(user["_id"])})
    cart["_id"] = str(cart.get("_id"))
    return cart


@app.post("/cart/items")
def add_to_cart(body: AddCartItemRequest, user=Depends(get_current_user)):
    cart = db["cart"].find_one({"user_id": str(user["_id"])})
    if not cart:
        cart = CartSchema(user_id=str(user["_id"]))
        create_document("cart", cart)
        cart = db["cart"].find_one({"user_id": str(user["_id"])})
    items: List[Dict[str, Any]] = cart.get("items", [])
    merged = False
    for it in items:
        if it["variant_sku"] == body.variant_sku and it.get("design_id") == body.design_id:
            it["quantity"] += body.quantity
            merged = True
            break
    if not merged:
        items.append(body.model_dump())
    subtotal = sum(i["price"] * i["quantity"] for i in items)
    cart_update = {"items": items, "subtotal": subtotal, "total": subtotal}
    db["cart"].update_one({"_id": cart["_id"]}, {"$set": cart_update})
    updated = db["cart"].find_one({"_id": cart["_id"]})
    updated["_id"] = str(updated.get("_id"))
    return updated


# ---- Uploads ----
ALLOWED_MIME = {"image/png": "png", "image/jpeg": "jpg", "image/svg+xml": "svg"}


@app.post("/uploads/images")
def upload_image(file: UploadFile = File(...), user=Depends(get_current_user)):
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    data = file.file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")
    ext = ALLOWED_MIME[file.content_type]
    fname = f"{datetime.now().timestamp():.0f}_{user['_id']}.{ext}"
    fpath = os.path.join(ASSETS_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(data)
    url = f"/assets/{fname}"
    create_document("upload", {"user_id": str(user["_id"]), "path": fpath, "url": url, "content_type": file.content_type})
    return {"url": url, "name": fname}


@app.get("/assets/{filename}")
def serve_asset(filename: str):
    fpath = os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(fpath)


# ---- Design render ----
class RenderRequest(BaseModel):
    design: DesignSchema


def _render_png(design: DesignSchema, out_path: str):
    img = Image.new("RGBA", (design.canvas_width, design.canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for layer in design.layers:
        ltype = layer.get("type") if isinstance(layer, dict) else layer.type
        if ltype == "text":
            l = layer if isinstance(layer, dict) else layer.model_dump()
            text = l.get("text")
            x = int(l.get("x")); y = int(l.get("y"))
            fill = l.get("fill", "#000")
            size = int(l.get("fontSize", 16))
            try:
                font = ImageFont.truetype("arial.ttf", size=size)
            except Exception:
                font = ImageFont.load_default()
            draw.text((x, y), text, fill=fill, font=font)
        elif ltype == "image":
            l = layer if isinstance(layer, dict) else layer.model_dump()
            src = l.get("src", "")
            fname = src.split("/")[-1]
            fpath = os.path.join(ASSETS_DIR, fname)
            if os.path.exists(fpath):
                layer_img = Image.open(fpath).convert("RGBA")
                sx = float(l.get("scaleX", 1)); sy = float(l.get("scaleY", 1))
                new_size = (max(1, int(layer_img.width * sx)), max(1, int(layer_img.height * sy)))
                layer_img = layer_img.resize(new_size)
                x = int(l.get("x", 0)); y = int(l.get("y", 0))
                img.alpha_composite(layer_img, (x, y))
    img.save(out_path, "PNG")


def _render_pdf(design: DesignSchema, out_path: str):
    c = pdf_canvas.Canvas(out_path, pagesize=letter)
    for layer in design.layers:
        ltype = layer.get("type") if isinstance(layer, dict) else layer.type
        if ltype == "text":
            l = layer if isinstance(layer, dict) else layer.model_dump()
            text = l.get("text"); x = float(l.get("x")); y = float(l.get("y"))
            c.setFillColorRGB(0, 0, 0)
            c.drawString(x, y, text)
        elif ltype == "image":
            l = layer if isinstance(layer, dict) else layer.model_dump()
            src = l.get("src", ""); fname = src.split("/")[-1]
            fpath = os.path.join(ASSETS_DIR, fname)
            if os.path.exists(fpath):
                x = float(l.get("x", 0)); y = float(l.get("y", 0))
                c.drawImage(fpath, x, y)
    c.showPage(); c.save()


@app.post("/designs/render")
def render_design(req: RenderRequest, user=Depends(get_current_user)):
    design = req.design
    now_tag = f"{int(datetime.now().timestamp())}"
    png_name = f"design_{now_tag}.png"; pdf_name = f"design_{now_tag}.pdf"
    png_path = os.path.join(ASSETS_DIR, png_name); pdf_path = os.path.join(ASSETS_DIR, pdf_name)
    _render_png(design, png_path); _render_pdf(design, pdf_path)
    design_doc = design.model_dump()
    design_doc.update({
        "user_id": str(user["_id"]),
        "asset_png_url": f"/assets/{png_name}",
        "asset_pdf_url": f"/assets/{pdf_name}",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    })
    did = create_document("design", design_doc)
    return {"id": did, "asset_png_url": design_doc["asset_png_url"], "asset_pdf_url": design_doc["asset_pdf_url"]}


# ---- Orders ----
@app.post("/orders")
def create_order(user=Depends(get_current_user)):
    cart = db["cart"].find_one({"user_id": str(user["_id"])})
    if not cart or not cart.get("items"):
        raise HTTPException(status_code=400, detail="Cart is empty")
    subtotal = sum(i["price"] * i["quantity"] for i in cart["items"])
    order = {
        "user_id": str(user["_id"]),
        "items": cart["items"],
        "currency": cart.get("currency", "USD"),
        "subtotal": subtotal,
        "discounts": 0,
        "tax": 0,
        "shipping": 0,
        "total": subtotal,
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    oid = create_document("order", order)
    return {"id": oid, "status": "pending", "total": order["total"]}


@app.get("/orders")
def list_orders(user=Depends(get_current_user)):
    orders = db["order"].find({"user_id": str(user["_id"])})
    items = []
    for o in orders:
        o["_id"] = str(o.get("_id"))
        items.append(o)
    return {"items": items}


@app.get("/admin/orders")
def admin_orders(user=Depends(get_current_user)):
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin only")
    items = []
    for o in db["order"].find({}).sort("created_at", -1):
        o["_id"] = str(o.get("_id")); items.append(o)
    return {"items": items}


@app.patch("/admin/orders/{order_id}/status")
def admin_update_order(order_id: str, body: UpdateOrderStatusRequest, user=Depends(get_current_user)):
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin only")
    from bson import ObjectId
    try:
        oid = ObjectId(order_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Order not found")
    db["order"].update_one({"_id": oid}, {"$set": {"status": body.status, "updated_at": datetime.now(timezone.utc)}})
    return {"ok": True}


@app.get("/admin/orders/export.csv")
def export_orders_csv(user=Depends(get_current_user)):
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin only")
    import csv
    tmp = os.path.join(ASSETS_DIR, f"orders_{int(datetime.now().timestamp())}.csv")
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "user_id", "status", "total", "created_at"])
        for o in db["order"].find({}):
            writer.writerow([str(o.get("_id")), o.get("user_id"), o.get("status"), o.get("total"), o.get("created_at")])
    return FileResponse(tmp, filename=os.path.basename(tmp))


# ---- Payments (Stripe) ----
@app.post("/payments/stripe/checkout")
def create_checkout_session(user=Depends(get_current_user)):
    import stripe
    secret = os.getenv("STRIPE_SECRET_KEY")
    if not secret:
        raise HTTPException(status_code=400, detail="Stripe not configured")
    stripe.api_key = secret
    cart = db["cart"].find_one({"user_id": str(user["_id"])})
    if not cart or not cart.get("items"):
        raise HTTPException(status_code=400, detail="Cart is empty")
    line_items = []
    for it in cart["items"]:
        line_items.append({
            "price_data": {
                "currency": cart.get("currency", "usd").lower(),
                "product_data": {"name": it.get("variant_sku", "Item")},
                "unit_amount": int(float(it["price"]) * 100),
            },
            "quantity": int(it["quantity"]),
        })
    domain = os.getenv("PUBLIC_URL", "http://localhost:3000")
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        mode="payment",
        line_items=line_items,
        success_url=f"{domain}/checkout/success?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{domain}/checkout/cancel",
        metadata={"user_id": str(user["_id"])},
    )
    return {"id": session.id, "url": session.url}


@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    import stripe
    if endpoint_secret:
        try:
            event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid webhook")
    else:
        event = await request.json()
    event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
    if event_type == "checkout.session.completed":
        sess = event["data"]["object"]
        uid = sess.get("metadata", {}).get("user_id")
        order = db["order"].find_one({"user_id": uid, "status": "pending"}, sort=[("created_at", -1)])
        if order:
            db["order"].update_one({"_id": order["_id"]}, {"$set": {"status": "paid", "payment_provider": "stripe", "payment_ref": sess.get("id")}})
    return {"received": True}


# ---- Promos ----
@app.post("/admin/promos")
def create_promo(promo: PromoSchema, user=Depends(get_current_user)):
    if "admin" not in user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin only")
    pid = create_document("promo", promo)
    return {"id": pid}


@app.get("/promos/validate")
def validate_promo(code: str):
    now = datetime.now(timezone.utc)
    p = db["promo"].find_one({"code": code, "active": True})
    if not p:
        raise HTTPException(status_code=404, detail="Invalid code")
    # optional window checks
    if p.get("starts_at") and now < p["starts_at"]:
        raise HTTPException(status_code=400, detail="Not active yet")
    if p.get("ends_at") and now > p["ends_at"]:
        raise HTTPException(status_code=400, detail="Expired")
    p["_id"] = str(p.get("_id"))
    return p


# ---- Health / diagnostics ----
@app.get("/")
def root():
    return {"message": "Ecommerce API running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        from database import db as _db
        if _db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = _db.name if hasattr(_db, 'name') else "✅ Connected"
            try:
                collections = _db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
