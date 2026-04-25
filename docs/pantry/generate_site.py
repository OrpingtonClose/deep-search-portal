#!/usr/bin/env python3
"""Generate the static pantry food gallery site."""
import os, re, html, json, textwrap

PANTRY_DIR = os.path.dirname(os.path.abspath(__file__))
ITEMS_DIR = os.path.join(PANTRY_DIR, "items")
os.makedirs(ITEMS_DIR, exist_ok=True)

# ── Master food data ────────────────────────────────────────────────
# Each entry: (slug, display_name, category, image_file, brief_desc)
# image_file maps to the best photo from the Grok conversation
# Categories match the markdown guide sections

CATEGORIES = [
    ("spanish", "Spanish Conservas & Premium Tinned Seafood"),
    ("nordic", "Nordic & Scandinavian Specialties"),
    ("polish", "Polish / Eastern European"),
    ("french", "French Products"),
    ("charcuterie", "Charcuterie, Sauces, Pickles & Other"),
]

FOODS = [
    # ── Spanish Conservas ──
    ("bluefin-tuna", "Herpac Bluefin Tuna Loin", "spanish", "batch2_03.jpg",
     "Almadraba-caught Atlantic bluefin tuna loin in Arbequina EVOO from Barbate, Cádiz."),
    ("mackerel-roes", "Herpac Mackerel Roes", "spanish", "batch3_01.jpg",
     "Atlantic chub mackerel roe sacs in extra-virgin olive oil. Briny, creamy, delicate."),
    ("sea-urchin-roe", "Sea Urchin Roe", "spanish", "batch1_02.jpg",
     "Galician sea urchin gonads — multiple brands: La Brújula Nº91, Ramón Peña 1920, Rosa Lafuente, Los Peperetes, Porto-Muiños."),
    ("stuffed-baby-squid", "Ramón Peña Stuffed Baby Squid", "spanish", "batch2_06.jpg",
     "Chipirones rellenos in their own ink sauce. Classic Galician conserva."),
    ("razor-clams", "Razor Clams / Navajas in Brine", "spanish", "batch1_10.jpg",
     "Galician razor clams preserved in clean brine. Multiple brands including Selección 1920."),
    ("mussels", "Mussels", "spanish", "batch3_07.jpg",
     "Mytilus galloprovincialis + Porto-Muiños Mejillón al Natural con Tronco de Wakame."),
    ("oysters", "Oysters in Albariño Vinaigrette", "spanish", "batch1_12.jpg",
     "Large Galician oysters dressed in Albariño wine vinaigrette. Ready-to-eat luxury."),
    ("octopus-cream", "Sotavento Octopus Cream", "spanish", "batch1_03.jpg",
     "Crema de Pulpo — smooth, spreadable octopus pâté from Spain."),
    ("spider-crab", "Spider Crab Meat", "spanish", "batch1_16.jpg",
     "Carne de Centollo al Natural from La Mar de Tazones, Asturias."),
    ("snow-crab", "Snow Crab Meat", "spanish", "batch2_07.jpg",
     "Cangrejo de las Nieves al Natural — multiple premium Spanish brands."),
    ("red-king-crab", "Red King Crab Leg Meat", "spanish", "batch2_12.jpg",
     "Chatka Patas 100% — red king crab leg meat in brine."),
    ("sardines", "Ferromar Small Sardines", "spanish", "batch2_13.jpg",
     "Sardinillas en Aceite de Oliva — small sardines in premium olive oil."),
    ("baby-eels", "Baby Eels (Angulas)", "spanish", "batch1_17.jpg",
     "Conservas de Cambados angulas in olive oil with guindilla chili."),
    ("sturgeon", "El Capricho Santoña Sturgeon", "spanish", "batch3_06.jpg",
     "Award-winning tinned sturgeon in olive oil from Santoña."),
    ("lobster-pate", "Agromar Lobster Pâté", "spanish", "batch3_02.jpg",
     "Paté de Bogavante — luxurious lobster rillettes in olive oil."),
    ("velvet-crab-cream", "La Venta Velvet Swimming Crab Cream", "spanish", "batch3_10.jpg",
     "Crema de Nécoras (1897) — velvet swimming crab cream from Cantabria."),

    # ── Nordic & Scandinavian ──
    ("limfjord-cockles", "Fangst Limfjord Cockles", "nordic", "batch2_01.jpg",
     "Danish heart cockles from the Limfjord, lightly salted. Sweet-briny perfection."),
    ("monkfish-liver", "Fangst Icelandic Monkfish Liver", "nordic", "batch2_08.jpg",
     "Havtaske Lever — lightly smoked Icelandic monkfish liver. 'Foie gras of the sea.'"),
    ("cod-liver-nordic", "Fangst Icelandic Cod Liver", "nordic", "batch3_04.jpg",
     "Torske Lever — lightly smoked Icelandic cod liver in own oil."),
    ("surstromming", "Röda Ulven Surströmming", "nordic", "batch1_08.jpg",
     "Famous Swedish fermented Baltic herring — the world's most pungent delicacy."),
    ("pickled-fennel", "Pickled Fennel with Dill", "nordic", "batch2_18.jpg",
     "Fennikel Syltet — Scandinavian-style preserved fennel with dill."),

    # ── Polish / Eastern European ──
    ("catfish-bbq", "Łosoś Ustka Catfish in Barbecue Sauce", "polish", "batch1_01.jpg",
     "Polish wels catfish (sum) in barbecue sauce. Ready-to-eat canned fish."),
    ("pickled-herring", "Kiszone Śledzie Łódzkie", "polish", "batch1_07.jpg",
     "Polish pickled/fermented herring fillets, Łódź-style."),
    ("cod-liver-riga", "Riga Gold Cod Liver", "polish", "batch2_02.jpg",
     "Wątróbki z dorsza w tłuszczu własnym — cod liver in own fat. Latvian brand, Icelandic production."),

    # ── French Products ──
    ("thon-basquaise", "Jean de Luz Basque Tuna", "french", "batch2_16.jpg",
     "Thon Basquaise aux Légumes Bio — albacore with organic vegetables and Espelette pepper."),
    ("monkfish-fillets", "Jean de Luz Monkfish Fillets", "french", "batch2_17.jpg",
     "Filet de Lotte au Naturel — monkfish in lightly salted spring water."),
    ("pork-ratatouille", "Maison Rivière Pork with Ratatouille", "french", "batch1_15.jpg",
     "Sauté de porc et sa ratatouille — classic French canned ready-meal stew."),
    ("pork-terrine", "Charcuterie Bobosse Pork Terrine", "french", "batch2_10.jpg",
     "Terrine Bobossienne N°5 — artisanal French pork terrine."),
    ("pork-liver-pate", "Pork Liver Pâté with Guérande Salt", "french", "batch3_05.jpg",
     "Pasztet z wątróbki z soli z Guérande — Franco-Polish pork liver pâté."),

    # ── Charcuterie, Sauces, Pickles & Other ──
    ("sobrasada", "El Zagal Sobrasada de Mallorca", "charcuterie", "batch1_06.jpg",
     "Traditional Mallorcan sobrasada — soft, spreadable cured pork with paprika."),
    ("baby-broad-beans", "Baby Broad Beans in Olive Oil", "charcuterie", "batch1_14.jpg",
     "Habitas baby en aceite de oliva — tender Spanish legumes, perfect as a tapa."),
    ("red-currant-jam", "Red Currant Jam", "charcuterie", "batch1_04.jpg",
     "Konfitura z czerwonych porzeczek — hand-labeled Polish fruit preserve."),
    ("chimichurri", "Artisanal Chimichurri Sauce", "charcuterie", "batch2_04.jpg",
     "Argentine-style chimichurri — parsley, chili, garlic, oregano, EVOO, wine vinegar."),
    ("tamarind-chutney", "Shan Tamarind Chutney", "charcuterie", "batch1_18.jpg",
     "Pakistani/Indian-style sweet-sour tamarind chutney. Tangy condiment."),
    ("mango-pickle", "Shan Mango & Karela Pickle", "charcuterie", "batch2_05.jpg",
     "Pakistani/Indian-style pickle of mango and bitter melon (karela) in spiced oil."),
    ("kimchi", "Jongga Napa Cabbage Kimchi", "charcuterie", "batch3_09.jpg",
     "Classic Korean napa cabbage kimchi. Spicy, fermented, umami-rich."),
]

# ── Read the full markdown guide ────────────────────────────────────
GUIDE_PATH = os.environ.get(
    "PANTRY_GUIDE_PATH",
    os.path.join(PANTRY_DIR, "gourmet_tinned_goods_guide.md"),
)

with open(GUIDE_PATH, "r", encoding="utf-8") as f:
    guide_text = f.read()

# Split guide into sections by ## numbered headers AND # category headers
sections = re.split(r'\n(?=## \d|# [A-Z])', guide_text)

# Build a lookup: slug -> markdown content
# We'll match food items to their guide sections
SLUG_TO_SECTION_MAP = {
    "bluefin-tuna": "## 1. Herpac Bluefin Tuna",
    "mackerel-roes": "## 2. Herpac Mackerel Roes",
    "sea-urchin-roe": "## 3. Sea Urchin Roe",
    "stuffed-baby-squid": "## 4. Ramón Peña Stuffed Baby Squid",
    "razor-clams": "## 5. Razor Clams",
    "mussels": "## 6. Mussels",
    "oysters": "## 7. Oysters",
    "octopus-cream": "## 8. Sotavento Octopus Cream",
    "spider-crab": "## 9. Spider Crab",
    "snow-crab": "## 10. Snow Crab",
    "red-king-crab": "## 11. Red King Crab",
    "sardines": "## 12. Ferromar Small Sardines",
    "baby-eels": "## 13. Baby Eels",
    "sturgeon": "## 14. El Capricho",
    "lobster-pate": "## 15. Agromar Lobster",
    "velvet-crab-cream": "## 16. La Venta",
}

# For Nordic items, they use ### headers
NORDIC_SECTION_MARKERS = {
    "limfjord-cockles": "Fangst Limfjord Cockles",
    "monkfish-liver": "Fangst Icelandic Monkfish Liver",
    "cod-liver-nordic": "Fangst Icelandic Cod Liver",
    "surstromming": "Surströmming",
    "pickled-fennel": "Pickled Fennel",
}

POLISH_SECTION_MARKERS = {
    "catfish-bbq": "## 1. Łosoś Ustka",
    "pickled-herring": "## 2. Kiszone",
    "cod-liver-riga": "## 3. Riga Gold",
}

FRENCH_SECTION_MARKERS = {
    "thon-basquaise": "Jean de Luz Basque Tuna",
    "monkfish-fillets": "Jean de Luz Monkfish Fillets",
    "pork-ratatouille": "Maison Rivière",
    "pork-terrine": "Charcuterie Bobosse",
    "pork-liver-pate": "Pork Liver Pâté",
}

CHARCUTERIE_SECTION_MARKERS = {
    "sobrasada": "El Zagal Sobrasada",
    "baby-broad-beans": "Baby Broad Beans",
    "red-currant-jam": "### Red Currant Jam",
    "chimichurri": "Artisanal Chimichurri",
    "tamarind-chutney": "Shan Tamarind Chutney",
    "mango-pickle": "Shan Mango",
    "kimchi": "Korean Napa Cabbage Kimchi",
}


ALL_MARKERS = {}
ALL_MARKERS.update(SLUG_TO_SECTION_MAP)
ALL_MARKERS.update(NORDIC_SECTION_MARKERS)
ALL_MARKERS.update(POLISH_SECTION_MARKERS)
ALL_MARKERS.update(FRENCH_SECTION_MARKERS)
ALL_MARKERS.update(CHARCUTERIE_SECTION_MARKERS)

# Pre-compute header line positions for reliable matching
_header_positions = []
for m in re.finditer(r'^(#{1,4}) .+', guide_text, re.MULTILINE):
    _header_positions.append((m.start(), m.group()))


def _find_header_pos(marker_text):
    """Find the position of a header line containing marker_text."""
    needle = marker_text.lower()
    for pos, header_line in _header_positions:
        if needle in header_line.lower():
            return pos
    return -1


def _extract_section(start_pos):
    """Extract text from start_pos to the next header of equal or higher level."""
    header_line_end = guide_text.index('\n', start_pos)
    header_line = guide_text[start_pos:header_line_end]
    level = len(header_line) - len(header_line.lstrip('#'))
    for pos, hdr in _header_positions:
        if pos <= start_pos:
            continue
        hdr_level = len(hdr) - len(hdr.lstrip('#'))
        if hdr_level <= level:
            return guide_text[start_pos:pos].strip()
    return guide_text[start_pos:].strip()


def find_section(slug):
    """Find the guide section for a food item."""
    marker = ALL_MARKERS.get(slug)
    if not marker:
        return None
    pos = _find_header_pos(marker.lstrip('#').strip())
    if pos < 0:
        return None
    return _extract_section(pos)


def md_to_html(md_text):
    """Very simple markdown-to-HTML converter for the guide sections."""
    if not md_text:
        return "<p><em>Detailed guide entry not available for this item.</em></p>"

    lines = md_text.split('\n')
    html_parts = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        # Skip top-level ## header (we handle that in the page)
        if stripped.startswith('## ') or stripped.startswith('### ') or stripped.startswith('#### '):
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            # Sub-headers within sections
            if stripped.startswith('#### '):
                html_parts.append(f'<h4>{html.escape(stripped[5:])}</h4>')
            elif stripped.startswith('### '):
                html_parts.append(f'<h3>{html.escape(stripped[4:])}</h3>')
            continue

        if stripped.startswith('---'):
            continue

        if stripped.startswith('- **') or stripped.startswith('- '):
            if not in_list:
                html_parts.append('<ul>')
                in_list = True
            content = html.escape(stripped[2:])
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
            html_parts.append(f'<li>{content}</li>')
        elif stripped == '':
            if in_list:
                html_parts.append('</ul>')
                in_list = False
        else:
            if in_list:
                html_parts.append('</ul>')
                in_list = False
            content = html.escape(stripped)
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
            html_parts.append(f'<p>{content}</p>')

    if in_list:
        html_parts.append('</ul>')

    return '\n'.join(html_parts)


# ── CSS ─────────────────────────────────────────────────────────────
CSS = """
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --border: #30363d;
  --text: #e6edf3;
  --text-muted: #8b949e;
  --accent: #c9a84c;
  --accent-dim: #8b7328;
  --link: #c9a84c;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Georgia', 'Times New Roman', serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.7;
  min-height: 100vh;
}
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }

.container { max-width: 1200px; margin: 0 auto; padding: 0 24px; }

/* Header */
header {
  border-bottom: 1px solid var(--border);
  padding: 24px 0;
  background: var(--surface);
}
header h1 {
  font-size: 1.8rem;
  color: var(--accent);
  font-weight: 400;
  letter-spacing: 0.02em;
}
header .subtitle {
  color: var(--text-muted);
  font-size: 0.95rem;
  margin-top: 4px;
  font-style: italic;
}

/* Breadcrumb */
.breadcrumb {
  padding: 16px 0;
  font-size: 0.85rem;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border);
}
.breadcrumb a { color: var(--text-muted); }
.breadcrumb a:hover { color: var(--accent); }
.breadcrumb .sep { margin: 0 8px; opacity: 0.5; }

/* Category section */
.category-section { margin: 40px 0; }
.category-section h2 {
  font-size: 1.3rem;
  color: var(--accent);
  font-weight: 400;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
}

/* Food grid */
.food-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 20px;
}
.food-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  transition: border-color 0.2s, transform 0.2s;
}
.food-card:hover {
  border-color: var(--accent-dim);
  transform: translateY(-2px);
}
.food-card a { color: inherit; display: block; }
.food-card a:hover { text-decoration: none; }
.food-card .card-img {
  width: 100%;
  aspect-ratio: 3/4;
  object-fit: cover;
  display: block;
  background: #1c2128;
}
.food-card .card-body {
  padding: 14px 16px;
}
.food-card .card-title {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 6px;
  line-height: 1.3;
}
.food-card .card-desc {
  font-size: 0.8rem;
  color: var(--text-muted);
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* Detail page */
.detail-hero {
  display: grid;
  grid-template-columns: 400px 1fr;
  gap: 40px;
  margin: 40px 0;
  align-items: start;
}
.detail-hero img {
  width: 100%;
  border-radius: 8px;
  border: 1px solid var(--border);
}
.detail-hero .hero-text h2 {
  font-size: 1.6rem;
  color: var(--accent);
  font-weight: 400;
  margin-bottom: 12px;
}
.detail-hero .hero-text .brief {
  font-size: 1rem;
  color: var(--text-muted);
  font-style: italic;
  margin-bottom: 20px;
  line-height: 1.6;
}
.detail-hero .hero-text .category-badge {
  display: inline-block;
  padding: 4px 12px;
  background: var(--accent-dim);
  color: var(--bg);
  font-size: 0.75rem;
  border-radius: 4px;
  font-weight: 600;
  letter-spacing: 0.03em;
  text-transform: uppercase;
}

.detail-content {
  max-width: 800px;
  margin: 0 auto 60px;
  padding: 0 24px;
}
.detail-content h3 {
  color: var(--accent);
  font-size: 1.15rem;
  font-weight: 400;
  margin: 32px 0 12px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}
.detail-content h4 {
  color: var(--accent);
  font-size: 1rem;
  font-weight: 600;
  margin: 24px 0 10px;
}
.detail-content p {
  margin: 12px 0;
  color: var(--text);
}
.detail-content ul {
  margin: 12px 0 12px 24px;
}
.detail-content li {
  margin: 6px 0;
  color: var(--text);
}
.detail-content strong { color: var(--text); }
.detail-content em { color: var(--text-muted); font-style: italic; }

/* Footer */
footer {
  border-top: 1px solid var(--border);
  padding: 24px 0;
  text-align: center;
  color: var(--text-muted);
  font-size: 0.8rem;
  background: var(--surface);
}

/* Responsive */
@media (max-width: 768px) {
  .detail-hero {
    grid-template-columns: 1fr;
  }
  .food-grid {
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 12px;
  }
  header h1 { font-size: 1.3rem; }
}
"""


def page_head(title, extra_css="", base_path="."):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<style>{CSS}{extra_css}</style>
</head>
<body>
"""

def page_header(base_path="."):
    return f"""
<header>
  <div class="container">
    <h1><a href="{base_path}/index.html" style="color:var(--accent)">The Private Chef's Pantry</a></h1>
    <div class="subtitle">A Curated Collection of Premium Tinned &amp; Jarred Goods</div>
  </div>
</header>
"""

def page_footer():
    return """
<footer>
  <div class="container">
    A Private Chef's Guide to Premium Tinned &amp; Jarred Goods
  </div>
</footer>
</body>
</html>
"""


# ── Generate index.html ─────────────────────────────────────────────
def generate_index():
    parts = [
        page_head("The Private Chef's Pantry"),
        page_header("."),
        '<div class="breadcrumb"><div class="container">',
        '<a href="./index.html">Pantry</a>',
        '</div></div>',
        '<main class="container">',
    ]

    for cat_slug, cat_name in CATEGORIES:
        cat_foods = [f for f in FOODS if f[2] == cat_slug]
        if not cat_foods:
            continue

        parts.append(f'<section class="category-section" id="{cat_slug}">')
        parts.append(f'<h2>{html.escape(cat_name)}</h2>')
        parts.append('<div class="food-grid">')

        for slug, name, _, img, brief in cat_foods:
            parts.append(f'''
<div class="food-card">
  <a href="items/{slug}.html">
    <img class="card-img" src="images/{img}" alt="{html.escape(name)}" loading="lazy">
    <div class="card-body">
      <div class="card-title">{html.escape(name)}</div>
      <div class="card-desc">{html.escape(brief)}</div>
    </div>
  </a>
</div>''')

        parts.append('</div>')
        parts.append('</section>')

    parts.append('</main>')
    parts.append(page_footer())

    with open(os.path.join(PANTRY_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write('\n'.join(parts))
    print("Generated index.html")


# ── Generate detail pages ───────────────────────────────────────────
def generate_detail_pages():
    for slug, name, cat_slug, img, brief in FOODS:
        cat_name = dict(CATEGORIES).get(cat_slug, "")
        section_md = find_section(slug)
        section_html = md_to_html(section_md)

        parts = [
            page_head(f"{name} — The Private Chef's Pantry", base_path=".."),
            page_header(".."),
            '<div class="breadcrumb"><div class="container">',
            '<a href="../index.html">Pantry</a>',
            f'<span class="sep">›</span>',
            f'<a href="../index.html#{cat_slug}">{html.escape(cat_name)}</a>',
            f'<span class="sep">›</span>',
            f'<span>{html.escape(name)}</span>',
            '</div></div>',
            '<main>',
            '<div class="container">',
            '<div class="detail-hero">',
            f'<img src="../images/{img}" alt="{html.escape(name)}">',
            '<div class="hero-text">',
            f'<span class="category-badge">{html.escape(cat_name)}</span>',
            f'<h2>{html.escape(name)}</h2>',
            f'<p class="brief">{html.escape(brief)}</p>',
            '</div>',
            '</div>',
            '</div>',
            '<div class="detail-content">',
            section_html,
            '</div>',
            '</main>',
            page_footer(),
        ]

        filepath = os.path.join(ITEMS_DIR, f"{slug}.html")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write('\n'.join(parts))
    print(f"Generated {len(FOODS)} detail pages")


if __name__ == "__main__":
    generate_index()
    generate_detail_pages()
    print("Done!")
