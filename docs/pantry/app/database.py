#!/usr/bin/env python3
"""SQLite database setup and seed data for the pantry app."""
import os
import re
import sqlite3
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PANTRY_DIR = APP_DIR.parent
DB_PATH = os.environ.get("PANTRY_DB_PATH", str(APP_DIR / "pantry.db"))

_conn = None


def get_db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
    return _conn


def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS foods (
            slug TEXT PRIMARY KEY,
            name_en TEXT NOT NULL,
            name_ru TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL,
            image TEXT NOT NULL,
            brief_en TEXT NOT NULL DEFAULT '',
            brief_ru TEXT NOT NULL DEFAULT '',
            guide_section_en TEXT DEFAULT '',
            guide_section_ru TEXT DEFAULT '',
            sort_order INTEGER NOT NULL DEFAULT 0
        );
    """)

    count = db.execute("SELECT COUNT(*) FROM foods").fetchone()[0]
    if count == 0:
        _seed_data(db)


# ── Guide section extraction ─────────────────────────────────────────

def _load_guide(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build_header_positions(text):
    positions = []
    for m in re.finditer(r'^(#{1,4}) .+', text, re.MULTILINE):
        positions.append((m.start(), m.group()))
    return positions


def _find_header_pos(positions, marker_text):
    needle = marker_text.lower()
    for pos, header_line in positions:
        if needle in header_line.lower():
            return pos
    return -1


def _extract_section(text, positions, start_pos):
    header_line_end = text.index('\n', start_pos)
    header_line = text[start_pos:header_line_end]
    level = len(header_line) - len(header_line.lstrip('#'))
    for pos, hdr in positions:
        if pos <= start_pos:
            continue
        hdr_level = len(hdr) - len(hdr.lstrip('#'))
        if hdr_level <= level:
            return text[start_pos:pos].strip()
    return text[start_pos:].strip()


def _find_section(text, positions, marker):
    clean_marker = marker.lstrip('#').strip()
    pos = _find_header_pos(positions, clean_marker)
    if pos < 0:
        return ""
    return _extract_section(text, positions, pos)


# ── Section markers (same as static generators) ──────────────────────

ALL_MARKERS = {
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
    "limfjord-cockles": "Fangst Limfjord Cockles",
    "monkfish-liver": "Fangst Icelandic Monkfish Liver",
    "cod-liver-nordic": "Fangst Icelandic Cod Liver",
    "surstromming": "Surströmming",
    "pickled-fennel": "Pickled Fennel",
    "catfish-bbq": "## 1. Łosoś Ustka",
    "pickled-herring": "## 2. Kiszone",
    "cod-liver-riga": "## 3. Riga Gold",
    "thon-basquaise": "Jean de Luz Basque Tuna",
    "monkfish-fillets": "Jean de Luz Monkfish Fillets",
    "pork-ratatouille": "Maison Rivière",
    "pork-terrine": "Charcuterie Bobosse",
    "pork-liver-pate": "Pork Liver Pâté",
    "sobrasada": "El Zagal Sobrasada",
    "baby-broad-beans": "Baby Broad Beans",
    "red-currant-jam": "### Red Currant Jam",
    "chimichurri": "Artisanal Chimichurri",
    "tamarind-chutney": "Shan Tamarind Chutney",
    "mango-pickle": "Shan Mango",
    "kimchi": "Korean Napa Cabbage Kimchi",
}

# Russian guide markers (similar patterns in the translated guide)
ALL_MARKERS_RU = {
    "bluefin-tuna": "## 1. Herpac",
    "mackerel-roes": "## 2. Herpac",
    "sea-urchin-roe": "## 3.",
    "stuffed-baby-squid": "## 4. Ramón Peña",
    "razor-clams": "## 5.",
    "mussels": "## 6.",
    "oysters": "## 7.",
    "octopus-cream": "## 8. Sotavento",
    "spider-crab": "## 9.",
    "snow-crab": "## 10.",
    "red-king-crab": "## 11.",
    "sardines": "## 12. Ferromar",
    "baby-eels": "## 13.",
    "sturgeon": "## 14. El Capricho",
    "lobster-pate": "## 15. Agromar",
    "velvet-crab-cream": "## 16. La Venta",
    "limfjord-cockles": "Fangst Limfjord",
    "monkfish-liver": "Fangst Icelandic Monkfish",
    "cod-liver-nordic": "Fangst Icelandic Cod",
    "surstromming": "Surströmming",
    "pickled-fennel": "Pickled Fennel",
    "catfish-bbq": "## 1. Łosoś Ustka",
    "pickled-herring": "## 2. Kiszone",
    "cod-liver-riga": "## 3. Riga Gold",
    "thon-basquaise": "Jean de Luz Basque",
    "monkfish-fillets": "Jean de Luz Monkfish",
    "pork-ratatouille": "Maison Rivière",
    "pork-terrine": "Charcuterie Bobosse",
    "pork-liver-pate": "Pork Liver",
    "sobrasada": "El Zagal Sobrasada",
    "baby-broad-beans": "Baby Broad",
    "red-currant-jam": "### Red Currant",
    "chimichurri": "Chimichurri",
    "tamarind-chutney": "Shan Tamarind",
    "mango-pickle": "Shan Mango",
    "kimchi": "Kimchi",
}


def _seed_data(db):
    """Seed the database with all food items and guide content."""

    # Load guides
    guide_en_path = PANTRY_DIR / "gourmet_tinned_goods_guide.md"
    guide_ru_path = PANTRY_DIR / "ru" / "gourmet_tinned_goods_guide_ru.md"
    guide_en = _load_guide(guide_en_path)
    guide_ru = _load_guide(guide_ru_path)
    positions_en = _build_header_positions(guide_en)
    positions_ru = _build_header_positions(guide_ru)

    FOODS = [
        # (slug, name_en, name_ru, category, image, brief_en, brief_ru)
        ("bluefin-tuna", "Herpac Bluefin Tuna Loin", "Herpac — филе голубого тунца", "spanish", "batch2_01.jpg",
         "Almadraba-caught Atlantic bluefin tuna loin in Arbequina EVOO from Barbate, Cádiz.",
         "Атлантический голубой тунец, пойманный способом альмадраба, филе в оливковом масле Arbequina EVOO из Барбате, Кадис."),
        ("mackerel-roes", "Herpac Mackerel Roes", "Herpac — икра макрели", "spanish", "batch3_05.jpg",
         "Atlantic chub mackerel roe sacs in extra-virgin olive oil. Briny, creamy, delicate.",
         "Икра атлантической скумбрии в оливковом масле. Солёная, кремовая, нежная."),
        ("sea-urchin-roe", "Sea Urchin Roe", "Икра морского ежа", "spanish", "batch1_01.jpg",
         "Galician sea urchin gonads — multiple brands: La Brújula Nº91, Ramón Peña 1920, Rosa Lafuente, Los Peperetes, Porto-Muiños.",
         "Галисийские гонады морского ежа — несколько брендов: La Brújula Nº91, Ramón Peña 1920, Rosa Lafuente, Los Peperetes, Porto-Muiños."),
        ("stuffed-baby-squid", "Ramón Peña Stuffed Baby Squid", "Ramón Peña — фаршированные кальмарчики", "spanish", "batch2_06.jpg",
         "Chipirones rellenos in their own ink sauce. Classic Galician conserva.",
         "Chipirones rellenos в собственных чернилах. Классическая галисийская консерва."),
        ("razor-clams", "Razor Clams / Navajas in Brine", "Морские черенки / Navajas в рассоле", "spanish", "batch1_10.jpg",
         "Galician razor clams preserved in clean brine. Multiple brands including Selección 1920.",
         "Галисийские морские черенки в чистом рассоле. Несколько брендов, включая Selección 1920."),
        ("mussels", "Mussels", "Мидии", "spanish", "batch1_12.jpg",
         "Mytilus galloprovincialis + Porto-Muiños Mejillón al Natural con Tronco de Wakame.",
         "Mytilus galloprovincialis + Porto-Muiños Mejillón al Natural con Tronco de Wakame."),
        ("oysters", "Oysters in Albariño Vinaigrette", "Устрицы в соусе винегрет из Альбариньо", "spanish", "batch1_09.jpg",
         "Large Galician oysters dressed in Albariño wine vinaigrette. Ready-to-eat luxury.",
         "Крупные галисийские устрицы в соусе на основе вина Альбариньо. Готовый к подаче деликатес."),
        ("octopus-cream", "Sotavento Octopus Cream", "Sotavento — крем из осьминога", "spanish", "batch1_07.jpg",
         "Crema de Pulpo — smooth, spreadable octopus pâté from Spain.",
         "Crema de Pulpo — нежный, намазываемый паштет из осьминога из Испании."),
        ("spider-crab", "Spider Crab Meat", "Мясо паука-краба", "spanish", "batch1_11.jpg",
         "Carne de Centollo al Natural from La Mar de Tazones, Asturias.",
         "Carne de Centollo al Natural от La Mar de Tazones, Астурия."),
        ("snow-crab", "Snow Crab Meat", "Мясо снежного краба", "spanish", "batch2_04.jpg",
         "Cangrejo de las Nieves al Natural — multiple premium Spanish brands.",
         "Cangrejo de las Nieves al Natural — несколько премиальных испанских брендов."),
        ("red-king-crab", "Red King Crab Leg Meat", "Мясо ног камчатского краба", "spanish", "batch2_12.jpg",
         "Chatka Patas 100% — red king crab leg meat in brine.",
         "Chatka Patas 100% — мясо ног камчатского краба в рассоле."),
        ("sardines", "Ferromar Small Sardines", "Ferromar — мелкие сардинки", "spanish", "batch2_11.jpg",
         "Sardinillas en Aceite de Oliva — small sardines in premium olive oil.",
         "Sardinillas en Aceite de Oliva — мелкие сардины в премиальном оливковом масле."),
        ("baby-eels", "Baby Eels (Angulas)", "Мальки угря (Angulas)", "spanish", "batch1_14.jpg",
         "Conservas de Cambados angulas in olive oil with guindilla chili.",
         "Conservas de Cambados angulas в оливковом масле с перцем гиндилья."),
        ("sturgeon", "El Capricho Santoña Sturgeon", "El Capricho Santoña — осётр", "spanish", "batch3_10.jpg",
         "Award-winning tinned sturgeon in olive oil from Santoña.",
         "Награждённый консервированный осётр в оливковом масле из Сантоньи."),
        ("lobster-pate", "Agromar Lobster Pâté", "Agromar — паштет из лобстера", "spanish", "batch3_03.jpg",
         "Paté de Bogavante — luxurious lobster rillettes in olive oil.",
         "Paté de Bogavante — роскошные рийеты из лобстера в оливковом масле."),
        ("velvet-crab-cream", "La Venta Velvet Swimming Crab Cream", "La Venta — крем из бархатного краба", "spanish", "batch3_07.jpg",
         "Crema de Nécoras (1897) — velvet swimming crab cream from Cantabria.",
         "Crema de Nécoras (1897) — крем из бархатного плавающего краба из Кантабрии."),

        # Nordic
        ("limfjord-cockles", "Fangst Limfjord Cockles", "Fangst — сердцевидки Лимфьорда", "nordic", "batch2_10.jpg",
         "Danish heart cockles from the Limfjord, lightly salted. Sweet-briny perfection.",
         "Датские сердцевидки из Лимфьорда, слегка подсоленные. Сладко-солёное совершенство."),
        ("monkfish-liver", "Fangst Icelandic Monkfish Liver", "Fangst — печень исландского морского чёрта", "nordic", "batch2_18.jpg",
         "Havtaske Lever — lightly smoked Icelandic monkfish liver. 'Foie gras of the sea.'",
         "Havtaske Lever — слегка копчёная печень исландского морского чёрта. «Фуа-гра моря»."),
        ("cod-liver-nordic", "Fangst Icelandic Cod Liver", "Fangst — печень исландской трески", "nordic", "batch3_06.jpg",
         "Torske Lever — lightly smoked Icelandic cod liver in own oil.",
         "Torske Lever — слегка копчёная печень исландской трески в собственном масле."),
        ("surstromming", "Röda Ulven Surströmming", "Röda Ulven Surströmming", "nordic", "batch1_08.jpg",
         "Famous Swedish fermented Baltic herring — the world's most pungent delicacy.",
         "Знаменитая шведская квашеная балтийская селёдка — самый ароматный деликатес в мире."),
        ("pickled-fennel", "Pickled Fennel with Dill", "Маринованный фенхель с укропом", "nordic", "batch2_15.jpg",
         "Fennikel Syltet — Scandinavian-style preserved fennel with dill.",
         "Fennikel Syltet — фенхель скандинавского засола с укропом."),

        # Polish
        ("catfish-bbq", "Łosoś Ustka Catfish in Barbecue Sauce", "Łosoś Ustka — сом в соусе барбекю", "polish", "batch1_03.jpg",
         "Polish wels catfish (sum) in barbecue sauce. Ready-to-eat canned fish.",
         "Польский сом (sum) в соусе барбекю. Готовая к употреблению рыбная консерва."),
        ("pickled-herring", "Kiszone Śledzie Łódzkie", "Kiszone Śledzie Łódzkie", "polish", "batch1_04.jpg",
         "Polish pickled/fermented herring fillets, Łódź-style.",
         "Польская маринованная/квашеная сельдь, по-лодзински."),
        ("cod-liver-riga", "Riga Gold Cod Liver", "Riga Gold — печень трески", "polish", "batch2_07.jpg",
         "Wątróbki z dorsza w tłuszczu własnym — cod liver in own fat. Latvian brand, Icelandic production.",
         "Wątróbki z dorsza w tłuszczu własnym — печень трески в собственном жире. Латвийский бренд, исландское производство."),

        # French
        ("thon-basquaise", "Jean de Luz Basque Tuna", "Jean de Luz — баскский тунец", "french", "batch2_13.jpg",
         "Thon Basquaise aux Légumes Bio — albacore with organic vegetables and Espelette pepper.",
         "Thon Basquaise aux Légumes Bio — альбакор с органическими овощами и перцем Эспелет."),
        ("monkfish-fillets", "Jean de Luz Monkfish Fillets", "Jean de Luz — филе морского чёрта", "french", "batch2_16.jpg",
         "Filet de Lotte au Naturel — monkfish in lightly salted spring water.",
         "Filet de Lotte au Naturel — морской чёрт в слегка подсоленной родниковой воде."),
        ("pork-ratatouille", "Maison Rivière Pork with Ratatouille", "Maison Rivière — свинина с рататуем", "french", "batch1_18.jpg",
         "Sauté de porc et sa ratatouille — classic French canned ready-meal stew.",
         "Sauté de porc et sa ratatouille — классическое французское тушёное блюдо в банке."),
        ("pork-terrine", "Charcuterie Bobosse Pork Terrine", "Charcuterie Bobosse — свиной террин", "french", "batch2_09.jpg",
         "Terrine Bobossienne N°5 — artisanal French pork terrine.",
         "Terrine Bobossienne N°5 — ремесленный французский свиной террин."),
        ("pork-liver-pate", "Pork Liver Pâté with Guérande Salt", "Паштет из свиной печени с солью Геранд", "french", "batch3_04.jpg",
         "Pasztet z wątróbki z soli z Guérande — Franco-Polish pork liver pâté.",
         "Pasztet z wątróbki z solą z Guérande — франко-польский паштет из свиной печени."),

        # Charcuterie
        ("sobrasada", "El Zagal Sobrasada de Mallorca", "El Zagal Sobrasada de Mallorca", "charcuterie", "batch1_05.jpg",
         "Traditional Mallorcan sobrasada — soft, spreadable cured pork with paprika.",
         "Традиционная мальоркинская собрасада — мягкая, намазываемая вяленая свинина с паприкой."),
        ("baby-broad-beans", "Baby Broad Beans in Olive Oil", "Молодые бобы в оливковом масле", "charcuterie", "batch1_13.jpg",
         "Habitas baby en aceite de oliva — tender Spanish legumes, perfect as a tapa.",
         "Habitas baby en aceite de oliva — нежные испанские бобы, идеальные как тапас."),
        ("red-currant-jam", "Red Currant Jam", "Варенье из красной смородины", "charcuterie", "batch1_02.jpg",
         "Konfitura z czerwonych porzeczek — hand-labeled Polish fruit preserve.",
         "Konfitura z czerwonych porzeczek — ручная работа, польское фруктовое варенье."),
        ("chimichurri", "Artisanal Chimichurri Sauce", "Домашний соус чимичурри", "charcuterie", "batch2_02.jpg",
         "Argentine-style chimichurri — parsley, chili, garlic, oregano, EVOO, wine vinegar.",
         "Аргентинский чимичурри — петрушка, чили, чеснок, орегано, оливковое масло, винный уксус."),
        ("tamarind-chutney", "Shan Tamarind Chutney", "Shan — тамариндовый чатни", "charcuterie", "batch1_17.jpg",
         "Pakistani/Indian-style sweet-sour tamarind chutney. Tangy condiment.",
         "Пакистанский/индийский сладко-кислый тамариндовый чатни. Острая приправа."),
        ("mango-pickle", "Shan Mango & Karela Pickle", "Shan — пикули из манго и карелы", "charcuterie", "batch2_05.jpg",
         "Pakistani/Indian-style pickle of mango and bitter melon (karela) in spiced oil.",
         "Пакистанские/индийские пикули из манго и горькой тыквы (карелы) в пряном масле."),
        ("kimchi", "Jongga Napa Cabbage Kimchi", "Jongga — кимчи из пекинской капусты", "charcuterie", "batch3_08.jpg",
         "Classic Korean napa cabbage kimchi. Spicy, fermented, umami-rich.",
         "Классическое корейское кимчи из пекинской капусты. Острое, ферментированное, богатое умами."),
    ]

    for i, (slug, name_en, name_ru, cat, img, brief_en, brief_ru) in enumerate(FOODS):
        # Extract guide sections
        marker_en = ALL_MARKERS.get(slug, "")
        section_en = _find_section(guide_en, positions_en, marker_en) if marker_en and guide_en else ""

        marker_ru = ALL_MARKERS_RU.get(slug, "")
        section_ru = _find_section(guide_ru, positions_ru, marker_ru) if marker_ru and guide_ru else ""

        db.execute(
            "INSERT INTO foods (slug, name_en, name_ru, category, image, brief_en, brief_ru, "
            "guide_section_en, guide_section_ru, sort_order) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (slug, name_en, name_ru, cat, img, brief_en, brief_ru, section_en, section_ru, i),
        )
    db.commit()
    print(f"Seeded {len(FOODS)} food items into database")
