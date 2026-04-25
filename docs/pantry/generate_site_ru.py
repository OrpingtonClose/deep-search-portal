#!/usr/bin/env python3
"""Generate the Russian static pantry food gallery site at /pantry/ru/."""
import os, re, html, json, textwrap

PANTRY_DIR = os.path.dirname(os.path.abspath(__file__))
RU_DIR = os.path.join(PANTRY_DIR, "ru")
ITEMS_DIR = os.path.join(RU_DIR, "items")
os.makedirs(ITEMS_DIR, exist_ok=True)

# ── Master food data (Russian) ──────────────────────────────────────
# Each entry: (slug, display_name, category, image_file, brief_desc)

CATEGORIES = [
    ("spanish", "Испанские консервы и премиум морепродукты"),
    ("nordic", "Скандинавские деликатесы"),
    ("polish", "Польские / Восточноевропейские"),
    ("french", "Французские продукты"),
    ("charcuterie", "Мясные деликатесы, соусы, соленья и прочее"),
]

FOODS = [
    # ── Испанские консервы ──
    ("bluefin-tuna", "Herpac — филе голубого тунца", "spanish", "batch2_01.jpg",
     "Атлантический голубой тунец, пойманный способом альмадраба, филе в оливковом масле Arbequina EVOO из Барбате, Кадис."),
    ("mackerel-roes", "Herpac — икра макрели", "spanish", "batch3_05.jpg",
     "Икра атлантической скумбрии в оливковом масле. Солёная, кремовая, нежная."),
    ("sea-urchin-roe", "Икра морского ежа", "spanish", "batch1_01.jpg",
     "Галисийские гонады морского ежа — несколько брендов: La Brújula Nº91, Ramón Peña 1920, Rosa Lafuente, Los Peperetes, Porto-Muiños."),
    ("stuffed-baby-squid", "Ramón Peña — фаршированные кальмарчики", "spanish", "batch2_06.jpg",
     "Chipirones rellenos в собственных чернилах. Классическая галисийская консерва."),
    ("razor-clams", "Морские черенки / Navajas в рассоле", "spanish", "batch1_10.jpg",
     "Галисийские морские черенки в чистом рассоле. Несколько брендов, включая Selección 1920."),
    ("mussels", "Мидии", "spanish", "batch1_12.jpg",
     "Mytilus galloprovincialis + Porto-Muiños Mejillón al Natural con Tronco de Wakame."),
    ("oysters", "Устрицы в соусе винегрет из Альбариньо", "spanish", "batch1_09.jpg",
     "Крупные галисийские устрицы в соусе на основе вина Альбариньо. Готовый к подаче деликатес."),
    ("octopus-cream", "Sotavento — крем из осьминога", "spanish", "batch1_07.jpg",
     "Crema de Pulpo — нежный, намазываемый паштет из осьминога из Испании."),
    ("spider-crab", "Мясо паука-краба", "spanish", "batch1_11.jpg",
     "Carne de Centollo al Natural от La Mar de Tazones, Астурия."),
    ("snow-crab", "Мясо снежного краба", "spanish", "batch2_04.jpg",
     "Cangrejo de las Nieves al Natural — несколько премиальных испанских брендов."),
    ("red-king-crab", "Мясо ног камчатского краба", "spanish", "batch2_12.jpg",
     "Chatka Patas 100% — мясо ног камчатского краба в рассоле."),
    ("sardines", "Ferromar — мелкие сардинки", "spanish", "batch2_11.jpg",
     "Sardinillas en Aceite de Oliva — мелкие сардины в премиальном оливковом масле."),
    ("baby-eels", "Мальки угря (Angulas)", "spanish", "batch1_14.jpg",
     "Conservas de Cambados angulas в оливковом масле с перцем гиндилья."),
    ("sturgeon", "El Capricho Santoña — осётр", "spanish", "batch3_10.jpg",
     "Награждённый консервированный осётр в оливковом масле из Сантоньи."),
    ("lobster-pate", "Agromar — паштет из лобстера", "spanish", "batch3_03.jpg",
     "Paté de Bogavante — роскошные рийеты из лобстера в оливковом масле."),
    ("velvet-crab-cream", "La Venta — крем из бархатного краба", "spanish", "batch3_07.jpg",
     "Crema de Nécoras (1897) — крем из бархатного плавающего краба из Кантабрии."),

    # ── Скандинавские ──
    ("limfjord-cockles", "Fangst — сердцевидки Лимфьорда", "nordic", "batch2_10.jpg",
     "Датские сердцевидки из Лимфьорда, слегка подсоленные. Сладко-солёное совершенство."),
    ("monkfish-liver", "Fangst — печень исландского морского чёрта", "nordic", "batch2_18.jpg",
     "Havtaske Lever — слегка копчёная печень исландского морского чёрта. «Фуа-гра моря»."),
    ("cod-liver-nordic", "Fangst — печень исландской трески", "nordic", "batch3_06.jpg",
     "Torske Lever — слегка копчёная печень исландской трески в собственном масле."),
    ("surstromming", "Röda Ulven Surströmming", "nordic", "batch1_08.jpg",
     "Знаменитая шведская квашеная балтийская селёдка — самый ароматный деликатес в мире."),
    ("pickled-fennel", "Маринованный фенхель с укропом", "nordic", "batch2_15.jpg",
     "Fennikel Syltet — фенхель скандинавского засола с укропом."),

    # ── Польские / Восточноевропейские ──
    ("catfish-bbq", "Łosoś Ustka — сом в соусе барбекю", "polish", "batch1_03.jpg",
     "Польский сом (sum) в соусе барбекю. Готовая к употреблению рыбная консерва."),
    ("pickled-herring", "Kiszone Śledzie Łódzkie", "polish", "batch1_04.jpg",
     "Польская маринованная/квашеная сельдь, по-лодзински."),
    ("cod-liver-riga", "Riga Gold — печень трески", "polish", "batch2_07.jpg",
     "Wątróbki z dorsza w tłuszczu własnym — печень трески в собственном жире. Латвийский бренд, исландское производство."),

    # ── Французские продукты ──
    ("thon-basquaise", "Jean de Luz — баскский тунец", "french", "batch2_13.jpg",
     "Thon Basquaise aux Légumes Bio — альбакор с органическими овощами и перцем Эспелет."),
    ("monkfish-fillets", "Jean de Luz — филе морского чёрта", "french", "batch2_16.jpg",
     "Filet de Lotte au Naturel — морской чёрт в слегка подсоленной родниковой воде."),
    ("pork-ratatouille", "Maison Rivière — свинина с рататуем", "french", "batch1_18.jpg",
     "Sauté de porc et sa ratatouille — классическое французское тушёное блюдо в банке."),
    ("pork-terrine", "Charcuterie Bobosse — свиной террин", "french", "batch2_09.jpg",
     "Terrine Bobossienne N°5 — ремесленный французский свиной террин."),
    ("pork-liver-pate", "Паштет из свиной печени с солью Геранд", "french", "batch3_04.jpg",
     "Pasztet z wątróbki z solą z Guérande — франко-польский паштет из свиной печени."),

    # ── Мясные деликатесы, соусы, соленья и прочее ──
    ("sobrasada", "El Zagal Sobrasada de Mallorca", "charcuterie", "batch1_05.jpg",
     "Традиционная мальоркинская собрасада — мягкая, намазываемая вяленая свинина с паприкой."),
    ("baby-broad-beans", "Молодые бобы в оливковом масле", "charcuterie", "batch1_13.jpg",
     "Habitas baby en aceite de oliva — нежные испанские бобы, идеальные как тапас."),
    ("red-currant-jam", "Варенье из красной смородины", "charcuterie", "batch1_02.jpg",
     "Konfitura z czerwonych porzeczek — ручная работа, польское фруктовое варенье."),
    ("chimichurri", "Домашний соус чимичурри", "charcuterie", "batch2_02.jpg",
     "Аргентинский чимичурри — петрушка, чили, чеснок, орегано, оливковое масло, винный уксус."),
    ("tamarind-chutney", "Shan — тамариндовый чатни", "charcuterie", "batch1_17.jpg",
     "Пакистанский/индийский сладко-кислый тамариндовый чатни. Острая приправа."),
    ("mango-pickle", "Shan — пикули из манго и карелы", "charcuterie", "batch2_05.jpg",
     "Пакистанские/индийские пикули из манго и горькой тыквы (карелы) в пряном масле."),
    ("kimchi", "Jongga — кимчи из пекинской капусты", "charcuterie", "batch3_08.jpg",
     "Классическое корейское кимчи из пекинской капусты. Острое, ферментированное, богатое умами."),
]

# ── Read the full Russian markdown guide ─────────────────────────────
GUIDE_PATH = os.environ.get(
    "PANTRY_GUIDE_PATH_RU",
    os.path.join(RU_DIR, "gourmet_tinned_goods_guide_ru.md"),
)

with open(GUIDE_PATH, "r", encoding="utf-8") as f:
    guide_text = f.read()

# Split guide into sections by ## numbered headers AND # category headers
sections = re.split(r'\n(?=## \d|# [А-ЯA-Z])', guide_text)

# Build section lookup using same slug-to-marker approach
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
        return "<p><em>Подробное описание для этого продукта пока недоступно.</em></p>"

    lines = md_text.split('\n')
    html_parts = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('## ') or stripped.startswith('### ') or stripped.startswith('#### '):
            if in_list:
                html_parts.append('</ul>')
                in_list = False
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

/* Language switcher */
.lang-switch {
  float: right;
  font-size: 0.85rem;
  margin-top: 4px;
}
.lang-switch a {
  color: var(--text-muted);
  padding: 2px 8px;
  border: 1px solid var(--border);
  border-radius: 4px;
  font-family: sans-serif;
}
.lang-switch a:hover { color: var(--accent); border-color: var(--accent-dim); text-decoration: none; }
.lang-switch a.active { color: var(--accent); border-color: var(--accent); }

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


def lang_switch_html(en_path, is_detail=False):
    return f'''<span class="lang-switch"><a href="{en_path}">EN</a> <a href="#" class="active">RU</a></span>'''


def page_head(title, extra_css="", base_path="."):
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<style>{CSS}{extra_css}</style>
</head>
<body>
"""

def page_header(base_path=".", en_index_path="../index.html"):
    return f"""
<header>
  <div class="container">
    {lang_switch_html(en_index_path)}
    <h1><a href="{base_path}/index.html" style="color:var(--accent)">Кладовая частного шефа</a></h1>
    <div class="subtitle">Кураторская коллекция премиальных консервов и деликатесов</div>
  </div>
</header>
"""

def page_footer():
    return """
<footer>
  <div class="container">
    Руководство частного шефа по премиум консервам и деликатесам
  </div>
</footer>
</body>
</html>
"""


# ── Generate index.html ─────────────────────────────────────────────
def generate_index():
    parts = [
        page_head("Кладовая частного шефа"),
        page_header(".", "../index.html"),
        '<div class="breadcrumb"><div class="container">',
        '<a href="./index.html">Кладовая</a>',
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
    <img class="card-img" src="../images/{img}" alt="{html.escape(name)}" loading="lazy">
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

    with open(os.path.join(RU_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write('\n'.join(parts))
    print("Generated ru/index.html")


# ── Generate detail pages ───────────────────────────────────────────
def generate_detail_pages():
    for slug, name, cat_slug, img, brief in FOODS:
        cat_name = dict(CATEGORIES).get(cat_slug, "")
        section_md = find_section(slug)
        section_html = md_to_html(section_md)

        en_detail_path = f"../../items/{slug}.html"

        parts = [
            page_head(f"{name} — Кладовая частного шефа", base_path=".."),
            page_header("..", en_detail_path),
            '<div class="breadcrumb"><div class="container">',
            '<a href="../index.html">Кладовая</a>',
            f'<span class="sep">›</span>',
            f'<a href="../index.html#{cat_slug}">{html.escape(cat_name)}</a>',
            f'<span class="sep">›</span>',
            f'<span>{html.escape(name)}</span>',
            '</div></div>',
            '<main>',
            '<div class="container">',
            '<div class="detail-hero">',
            f'<img src="../../images/{img}" alt="{html.escape(name)}">',
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
    print(f"Generated {len(FOODS)} Russian detail pages")


if __name__ == "__main__":
    generate_index()
    generate_detail_pages()
    print("Done! Russian site generated at ru/")
