# -*- coding: utf-8 -*-
"""
build_shapenet_index.py

ShapeNetSem'in metadata.csv dosyasını okuyup,
basit bir kategori -> fullId listesi index'i üretir.

Çıktı: shapenet_index.json

Örnek:
{
  "chair": ["wss.100f39", "wss.1022fe", ...],
  "table": ["wss.104221", ...],
  ...
}
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

# !!! BUNU KENDİ PATH'İNE GÖRE KONTROL ET !!!
# Şu an senin SS'e göre:
#   D:\Shapenet\ShapeNetSem-backup
SHAPENET_ROOT = Path(r"D:\Shapenet\ShapeNetSem-backup")

# ShapeNetSem kökünde duran metadata.csv
METADATA_CSV = SHAPENET_ROOT / "metadata.csv"

# Çıkacak index dosyası (ai_engine klasörüne)
OUT_JSON = Path("shapenet_index.json")

# Her kategori için en fazla kaç model saklayalım
MAX_PER_CATEGORY = 32

# Bizim sistemde kullanılan "kanonik" kategoriler ve
# bu kategoriyi işaret eden keyword'ler
TARGET_CATEGORIES: dict[str, list[str]] = {
    # core furniture
    "chair": [
        "chair", "armchair", "office chair", "side chair",
        "recliner", "stool", "chaise", "loveseat"
    ],
    "table": [
        "table", "coffee table", "end table",
        "dining table", "bar table", "round table"
    ],
    "desk": [
        "desk", "computer desk", "workstation"
    ],
    "lamp": [
        "lamp", "floor lamp", "table lamp", "desk lamp", "ceiling lamp"
    ],
    "bed": [
        "bed", "bunk bed", "loft bed", "single bed", "double bed",
        "queen bed", "king bed"
    ],
    "sofa": [
        "sofa", "couch", "sectional", "futon"
    ],
    "bookshelf": [
        "bookcase", "bookshelf", "shelf unit"
    ],
    "cabinet": [
        "cabinet", "cupboard", "sideboard", "dresser", "chest of drawers"
    ],
    "plant": [
        "plant", "potted plant"
    ],
    "tree": [
        "tree"
    ],
    "fence": [
        "fence", "railing"
    ],
    "road_segment": [
        "road", "street", "pavement"
    ],
    "tv": [
        "tv", "television", "monitor"
    ],

    # ek gündelik nesneler
    "book": [
        "book", "books"
    ],
    "pencil": [
        "pencil", "pen", "marker"
    ],
    "bottle": [
        "bottle", "wine bottle", "drink bottle"
    ],
    "cup": [
        "cup", "mug"
    ],
    "guitar": [
        "guitar"
    ],
    "phone": [
        "phone", "cellphone", "telephone"
    ],
    "keyboard": [
        "keyboard", "piano keyboard"
    ],
    "mouse": [
        "mouse", "computer mouse"
    ],
    "car": [
        "car"
    ],
    "airplane": [
        "airplane", "plane"
    ],
    "bus": [
        "bus"
    ],
}

# =========================================================
# HELPER FONKSİYONLAR
# =========================================================

def _norm(s: str | None) -> str:
    return (s or "").lower()


def _build_search_text(row: dict) -> str:
    """
    Bir satır için arama metni:
    category + wnlemmas + name + tags
    """
    parts: list[str] = []
    for key in ("category", "wnlemmas", "name", "tags"):
        if key in row and row[key]:
            parts.append(str(row[key]).lower())
    return " ".join(parts)


# =========================================================
# ANA FONKSİYON
# =========================================================

def build_index() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"metadata.csv bulunamadı: {METADATA_CSV}")

    print("[1] Reading metadata.csv from:", METADATA_CSV)

    with METADATA_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[1] Toplam satır: {len(rows)}")

    # Satırlara arama metni ekle
    for row in rows:
        row["_text"] = _build_search_text(row)

    index: dict[str, list[str]] = {}

    # Her hedef kategori için eşleşmeleri topla
    for canon_cat, keywords in TARGET_CATEGORIES.items():
        hits: list[str] = []

        for row in rows:
            text = row["_text"]
            if not text:
                continue

            if any(_norm(kw) in text for kw in keywords):
                full_id = row.get("fullId") or row.get("fullid") or row.get("full_id")
                if not full_id:
                    continue
                hits.append(full_id)

        random.shuffle(hits)
        chosen = hits[:MAX_PER_CATEGORY]
        index[canon_cat] = chosen

        print(f"[CAT] {canon_cat:12s} -> {len(chosen):3d} seçildi (toplam {len(hits):3d} eşleşme)")

    # JSON olarak kaydet
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print("\n[OK] Index yazıldı →", OUT_JSON.resolve())


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    build_index()
