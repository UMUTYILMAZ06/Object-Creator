# shapenet_picker.py
#
# Amaç:
#   data/shapenet_index.json dosyasını yükleyip,
#   istenen kategori için bir ShapeNet fullId seçmek.
#
# Kullanım:
#   picker = ShapeNetPicker()
#   full_id = picker.pick("chair")

import json
import random
from pathlib import Path

INDEX_PATH = Path("data/shapenet_index.json")

# Eğer NLU tarafında ek bir isim kullanırsan (bookshelf vs bookcase)
# burada canonical isimlere map edebilirsin.
CANONICAL_CATEGORY = {
    "bookshelf": "bookshelf",
    "bookcase": "bookshelf",
    "sofa": "sofa",
    "couch": "sofa",
    "roadsegment": "road_segment",
    "road_segment": "road_segment",
    # gerekirse artırabilirsin
}

class ShapeNetPicker:
    def __init__(self, index_path: Path | None = None):
        path = index_path or INDEX_PATH
        if not path.exists():
            print(f"[ShapeNetPicker] Index bulunamadı: {path}. (build_shapenet_index.py çalıştı mı?)")
            self.index = {}
            return

        with path.open("r", encoding="utf-8") as f:
            self.index = json.load(f)

        print(f"[ShapeNetPicker] Index yüklendi: {path}")

    def _canon(self, category: str | None) -> str | None:
        if not category:
            return None
        c = category.lower().strip()
        return CANONICAL_CATEGORY.get(c, c)

    def pick(self, category: str | None):
        """
        NLU kategorisi (chair, desk, lamp...) alır,
        index'ten bir fullId döndürür. Yoksa None.
        """
        if not category:
            return None

        cat = self._canon(category)
        models = self.index.get(cat)
        if not models:
            # hiç model yoksa None
            return None

        # Rastgele bir model seç (istersen deterministic için models[0] da yapabilirsin)
        return random.choice(models)
