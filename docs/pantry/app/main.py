#!/usr/bin/env python3
"""FastAPI pantry food gallery – database-backed replica of the static site."""
import os
import re
import html as html_mod
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database import get_db, init_db

APP_DIR = Path(__file__).resolve().parent
PANTRY_DIR = APP_DIR.parent
TEMPLATES_DIR = APP_DIR / "templates"


@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    yield


ROOT_PATH = os.environ.get("PANTRY_ROOT_PATH", "/pantry")

app = FastAPI(title="The Private Chef's Pantry", lifespan=lifespan, root_path=ROOT_PATH)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Serve images from the existing images directory
app.mount("/images", StaticFiles(directory=str(PANTRY_DIR / "images")), name="images")


# ── Helpers ──────────────────────────────────────────────────────────

CATEGORIES_EN = [
    ("spanish", "Spanish Conservas & Premium Tinned Seafood"),
    ("nordic", "Nordic & Scandinavian Specialties"),
    ("polish", "Polish / Eastern European"),
    ("french", "French Products"),
    ("charcuterie", "Charcuterie, Sauces, Pickles & Other"),
]

CATEGORIES_RU = [
    ("spanish", "Испанские консервы и премиум морепродукты"),
    ("nordic", "Скандинавские деликатесы"),
    ("polish", "Польские / Восточноевропейские"),
    ("french", "Французские продукты"),
    ("charcuterie", "Мясные деликатесы, соусы, соленья и прочее"),
]

UI_STRINGS = {
    "en": {
        "site_title": "The Private Chef's Pantry",
        "subtitle": "A Curated Collection of Premium Tinned & Jarred Goods",
        "breadcrumb_home": "Pantry",
        "footer": "A Private Chef's Guide to Premium Tinned & Jarred Goods",
        "no_description": "Detailed guide entry not available for this item.",
        "categories": dict(CATEGORIES_EN),
    },
    "ru": {
        "site_title": "Кладовая частного шефа",
        "subtitle": "Кураторская коллекция премиальных консервов и деликатесов",
        "breadcrumb_home": "Кладовая",
        "footer": "Путеводитель частного шефа по премиальным консервам и деликатесам",
        "no_description": "Подробное описание для этого продукта пока недоступно.",
        "categories": dict(CATEGORIES_RU),
    },
}


def md_to_html(md_text: str) -> str:
    """Simple markdown-to-HTML for guide sections."""
    if not md_text:
        return ""
    lines = md_text.split("\n")
    parts = []
    in_list = False
    for line in lines:
        s = line.strip()
        if s.startswith("## ") or s.startswith("### ") or s.startswith("#### "):
            if in_list:
                parts.append("</ul>")
                in_list = False
            if s.startswith("#### "):
                parts.append(f"<h4>{html_mod.escape(s[5:])}</h4>")
            elif s.startswith("### "):
                parts.append(f"<h3>{html_mod.escape(s[4:])}</h3>")
            continue
        if s.startswith("---"):
            continue
        if s.startswith("- "):
            if not in_list:
                parts.append("<ul>")
                in_list = True
            content = html_mod.escape(s[2:])
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"\*(.+?)\*", r"<em>\1</em>", content)
            parts.append(f"<li>{content}</li>")
        elif s == "":
            if in_list:
                parts.append("</ul>")
                in_list = False
        else:
            if in_list:
                parts.append("</ul>")
                in_list = False
            content = html_mod.escape(s)
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"\*(.+?)\*", r"<em>\1</em>", content)
            parts.append(f"<p>{content}</p>")
    if in_list:
        parts.append("</ul>")
    return "\n".join(parts)


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def index_en(request: Request):
    return await render_index(request, "en")


@app.get("/ru", response_class=HTMLResponse)
@app.get("/ru/", response_class=HTMLResponse)
@app.get("/ru/index.html", response_class=HTMLResponse)
async def index_ru(request: Request):
    return await render_index(request, "ru")


async def render_index(request: Request, lang: str):
    db = get_db()
    ui = UI_STRINGS[lang]
    categories = CATEGORIES_EN if lang == "en" else CATEGORIES_RU
    cat_foods = {}
    for cat_slug, _ in categories:
        rows = db.execute(
            "SELECT slug, name_en, name_ru, category, image, brief_en, brief_ru "
            "FROM foods WHERE category = ? ORDER BY sort_order",
            (cat_slug,),
        ).fetchall()
        if rows:
            cat_foods[cat_slug] = rows
    return templates.TemplateResponse(
        request,
        "index.html",
        context={
            "lang": lang,
            "ui": ui,
            "categories": categories,
            "cat_foods": cat_foods,
            "prefix": ROOT_PATH,
        },
    )


@app.get("/items/{slug}.html", response_class=HTMLResponse)
async def detail_en(request: Request, slug: str):
    return await render_detail(request, slug, "en")


@app.get("/ru/items/{slug}.html", response_class=HTMLResponse)
async def detail_ru(request: Request, slug: str):
    return await render_detail(request, slug, "ru")


async def render_detail(request: Request, slug: str, lang: str):
    db = get_db()
    ui = UI_STRINGS[lang]
    row = db.execute(
        "SELECT slug, name_en, name_ru, category, image, brief_en, brief_ru, "
        "guide_section_en, guide_section_ru FROM foods WHERE slug = ?",
        (slug,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Food item not found")

    name = row["name_ru"] if lang == "ru" else row["name_en"]
    brief = row["brief_ru"] if lang == "ru" else row["brief_en"]
    guide_md = row["guide_section_ru"] if lang == "ru" else row["guide_section_en"]
    guide_html = md_to_html(guide_md) if guide_md else ""
    cat_name = ui["categories"].get(row["category"], "")

    return templates.TemplateResponse(
        request,
        "detail.html",
        context={
            "lang": lang,
            "ui": ui,
            "slug": slug,
            "name": name,
            "brief": brief,
            "image": row["image"],
            "category_slug": row["category"],
            "category_name": cat_name,
            "guide_html": guide_html,
            "prefix": ROOT_PATH,
        },
    )


IMAGES_DIR = PANTRY_DIR / "images"

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

ALL_CATEGORIES = [
    ("spanish", "Spanish Conservas & Premium Tinned Seafood"),
    ("nordic", "Nordic & Scandinavian Specialties"),
    ("polish", "Polish / Eastern European"),
    ("french", "French Products"),
    ("charcuterie", "Charcuterie, Sauces, Pickles & Other"),
]


# ── Admin routes ─────────────────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse)
@app.get("/admin/", response_class=HTMLResponse)
async def admin_list(request: Request):
    db = get_db()
    foods = db.execute(
        "SELECT slug, name_en, name_ru, category, image, brief_en, sort_order "
        "FROM foods ORDER BY sort_order"
    ).fetchall()
    return templates.TemplateResponse(
        request,
        "admin.html",
        context={
            "foods": foods,
            "categories": ALL_CATEGORIES,
            "prefix": ROOT_PATH,
            "mode": "list",
        },
    )


@app.get("/admin/add", response_class=HTMLResponse)
async def admin_add_form(request: Request):
    images = sorted(f.name for f in IMAGES_DIR.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    return templates.TemplateResponse(
        request,
        "admin.html",
        context={
            "categories": ALL_CATEGORIES,
            "images": images,
            "prefix": ROOT_PATH,
            "mode": "add",
            "food": None,
        },
    )


@app.post("/admin/add")
async def admin_add_submit(
    request: Request,
    slug: str = Form(...),
    name_en: str = Form(...),
    name_ru: str = Form(""),
    category: str = Form(...),
    image: str = Form(""),
    brief_en: str = Form(""),
    brief_ru: str = Form(""),
    guide_section_en: str = Form(""),
    guide_section_ru: str = Form(""),
    uploaded_image: UploadFile = File(None),
):
    db = get_db()
    existing = db.execute("SELECT slug FROM foods WHERE slug = ?", (slug,)).fetchone()
    if existing:
        raise HTTPException(status_code=400, detail=f"Item with slug '{slug}' already exists")

    if uploaded_image and uploaded_image.filename:
        img_data = await uploaded_image.read()
        safe_filename = Path(uploaded_image.filename).name
        if not safe_filename or Path(safe_filename).suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Only .jpg, .jpeg, .png, .webp files are allowed")
        img_path = IMAGES_DIR / safe_filename
        img_path.write_bytes(img_data)
        image = safe_filename

    max_order = db.execute("SELECT COALESCE(MAX(sort_order), 0) FROM foods").fetchone()[0]

    db.execute(
        "INSERT INTO foods (slug, name_en, name_ru, category, image, brief_en, brief_ru, "
        "guide_section_en, guide_section_ru, sort_order) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (slug, name_en, name_ru, category, image, brief_en, brief_ru,
         guide_section_en, guide_section_ru, max_order + 1),
    )
    db.commit()
    return RedirectResponse(url=f"{ROOT_PATH}/admin", status_code=303)


@app.get("/admin/edit/{slug}", response_class=HTMLResponse)
async def admin_edit_form(request: Request, slug: str):
    db = get_db()
    food = db.execute("SELECT * FROM foods WHERE slug = ?", (slug,)).fetchone()
    if not food:
        raise HTTPException(status_code=404, detail="Food item not found")
    images = sorted(f.name for f in IMAGES_DIR.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    return templates.TemplateResponse(
        request,
        "admin.html",
        context={
            "food": dict(food),
            "categories": ALL_CATEGORIES,
            "images": images,
            "prefix": ROOT_PATH,
            "mode": "edit",
        },
    )


@app.post("/admin/edit/{slug}")
async def admin_edit_submit(
    request: Request,
    slug: str,
    name_en: str = Form(...),
    name_ru: str = Form(""),
    category: str = Form(...),
    image: str = Form(""),
    brief_en: str = Form(""),
    brief_ru: str = Form(""),
    guide_section_en: str = Form(""),
    guide_section_ru: str = Form(""),
    uploaded_image: UploadFile = File(None),
):
    db = get_db()
    food = db.execute("SELECT slug FROM foods WHERE slug = ?", (slug,)).fetchone()
    if not food:
        raise HTTPException(status_code=404, detail="Food item not found")

    if uploaded_image and uploaded_image.filename:
        img_data = await uploaded_image.read()
        safe_filename = Path(uploaded_image.filename).name
        if not safe_filename or Path(safe_filename).suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Only .jpg, .jpeg, .png, .webp files are allowed")
        img_path = IMAGES_DIR / safe_filename
        img_path.write_bytes(img_data)
        image = safe_filename

    db.execute(
        "UPDATE foods SET name_en=?, name_ru=?, category=?, image=?, brief_en=?, brief_ru=?, "
        "guide_section_en=?, guide_section_ru=? WHERE slug=?",
        (name_en, name_ru, category, image, brief_en, brief_ru,
         guide_section_en, guide_section_ru, slug),
    )
    db.commit()
    return RedirectResponse(url=f"{ROOT_PATH}/admin", status_code=303)


@app.post("/admin/delete/{slug}")
async def admin_delete(slug: str):
    db = get_db()
    db.execute("DELETE FROM foods WHERE slug = ?", (slug,))
    db.commit()
    return RedirectResponse(url=f"{ROOT_PATH}/admin", status_code=303)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8090, reload=True)
