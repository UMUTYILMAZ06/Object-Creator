# -*- coding: utf-8 -*-
# run_command_loop.py
#
# Amaç:
#  - intent_head_v2.pt + slot_head_v3.pt + (opsiyonel) RoleTagger (HF klasörü) modellerini yükler
#  - Konsoldan İngilizce cümle alır
#  - Intent + slot + (varsa) RoleTagger + ufak heuristiklerle komut JSON'u üretir
#  - JSON'u Unity'nin okuduğu next_command.json dosyasına yazar
#  - moved_category için metadata.csv içinden bir model fullId seçip JSON'a ekler
#
# EK:
#  - Object attribute HF modeli (AutoModelForSequenceClassification) yüklüyse
#    0.50 üstü attribute'ları args.object_attribute içine ekler.
#
# ÖNEMLİ:
#  - RelationHead (relation_head_v1.pt) yüklüyse SlotHead relation_logits OVERRIDE edilir.

import re
import json
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ======= Dosya yolları =======

INTENT_MODEL_PATH   = Path("intent_head_v2.pt")
SLOT_MODEL_PATH     = Path("slot_head_v3.pt")

# RoleTagger (span'ler için, HF klasörü)
ROLE_TAGGER_DIR     = Path("roletagger_bert_v1")

# Relation modeli (SBERT + MLP)
RELATION_MODEL_PATH = Path("relation_head_v1.pt")

# ShapeNet metadata.csv yolu
METADATA_CSV_PATH = Path(r"D:\Shapenet\ShapeNetSem-backup\metadata.csv")

# Unity'ye çıkacak komut
NEXT_COMMAND_PATH = Path(r"C:\Users\90553\MERGEN\Assets\StreamingAssets\next_command.json")

# Object attribute modeli (HF klasörü) -> SENİN KLASÖR
OBJECT_ATTR_DIR = Path(r"C:\Users\90553\MERGEN\ai_engine\ObjectType")

# Attribute isimleri (fallback)
ATTRS_FALLBACK = [
    "supports_on_top",
    "can_be_placed_on",
    "is_container_openable",
    "has_interior_volume",
    "can_go_under",
    "can_be_under",
    "is_wall_mounted",
    "is_floor_object",
    "is_hanging_object",
]

# ========= metadata.csv loader =========

METADATA_ROWS: list[dict] = []

if METADATA_CSV_PATH.exists():
    try:
        with METADATA_CSV_PATH.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            METADATA_ROWS = list(reader)
        print(f"[ShapeNet/metadata] Loaded {len(METADATA_ROWS)} rows from {METADATA_CSV_PATH}")
        print(f"[ShapeNet/metadata] Columns: {reader.fieldnames}")
    except Exception as e:
        print("[ShapeNet/metadata] ERROR while loading metadata.csv:", e)
        METADATA_ROWS = []
else:
    print(f"[ShapeNet/metadata] WARNING: metadata.csv not found at: {METADATA_CSV_PATH}")
    METADATA_ROWS = []


def lookup_fullid_from_metadata(category: str | None) -> str | None:
    if not category:
        print("[ShapeNet/metadata] No category/span given.")
        return None

    if not METADATA_ROWS:
        print("[ShapeNet/metadata] No metadata rows loaded.")
        return None

    cat = category.lower()
    candidates: list[tuple[int, dict]] = []

    for row in METADATA_ROWS:
        row_cat  = (row.get("category") or "").lower()
        wnlemmas = (row.get("wnlemmas") or "").lower()
        name     = (row.get("name") or "").lower()
        tags     = (row.get("tags") or "").lower()

        text_any = row_cat or wnlemmas or name or tags
        if not text_any:
            continue

        score = 0
        if cat in row_cat:
            score += 3
        if cat in wnlemmas:
            score += 2
        if cat in name:
            score += 2
        if cat in tags:
            score += 1

        if score > 0:
            candidates.append((score, row))

    if not candidates:
        print(f"[ShapeNet/metadata] No match found for '{category}'")
        return None

    max_score = max(s for s, _ in candidates)
    best_rows = [r for s, r in candidates if s == max_score]
    row = random.choice(best_rows)

    full_id = row.get("fullId") or row.get("fullid") or row.get("full_id")
    if not full_id:
        print(f"[ShapeNet/metadata] Row has no fullId for '{category}'")
        return None

    print(f"[ShapeNet/metadata] Picked fullId='{full_id}' for query='{category}' (score={max_score})")
    return full_id


def pick_shapenet_model(category: str | None) -> str | None:
    return lookup_fullid_from_metadata(category)


# ========= IntentHead =========

class IntentHead(nn.Module):
    def __init__(self, emb_dim: int, num_intents: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_intents),
        )

    def forward(self, x):
        return self.net(x)


# ========= RelationHead =========

class RelationHead(nn.Module):
    def __init__(self, emb_dim: int, num_relations: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_relations),
        )

    def forward(self, x):
        return self.net(x)


# ========= SlotHead (v3 uyumlu) =========

class SlotHeadV2(nn.Module):
    def __init__(self, emb_dim, num_cat, num_rel, num_side, num_qty):
        super().__init__()
        hidden_dim1 = 512
        hidden_dim2 = 256

        self.backbone = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.moved_category_head     = nn.Linear(hidden_dim2, num_cat)
        self.reference_category_head = nn.Linear(hidden_dim2, num_cat)
        self.relation_head           = nn.Linear(hidden_dim2, num_rel)
        self.side_head               = nn.Linear(hidden_dim2, num_side)
        self.quantity_head           = nn.Linear(hidden_dim2, num_qty)
        self.distance_head           = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        h = self.backbone(x)
        return {
            "moved_category_logits":     self.moved_category_head(h),
            "reference_category_logits": self.reference_category_head(h),
            "relation_logits":           self.relation_head(h),
            "side_logits":               self.side_head(h),
            "quantity_logits":           self.quantity_head(h),
            "distance":                  self.distance_head(h).squeeze(-1),
        }


# ========= Object Attribute prediction (HF) =========

def predict_object_attributes(
    obj_text: str,
    tokenizer,
    model,
    label_list: list[str],
    device,
    threshold: float = 0.50,
    max_len: int = 64,
) -> list[str]:
    """
    HF multi-label: logits -> sigmoid -> threshold
    """
    if (not obj_text) or tokenizer is None or model is None or not label_list:
        return []

    model.eval()
    enc = tokenizer(
        obj_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding="max_length",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        logits = out.logits              # [1, num_labels]
        probs = torch.sigmoid(logits)[0] # [num_labels]

    picked = []
    for i, p in enumerate(probs.tolist()):
        if i < len(label_list) and p >= threshold:
            picked.append(label_list[i])

    return picked


# ========= Quantity & distance heuristics =========

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

OBJECT_PATTERN = (
    r"(?:sofa|sofas|table|tables|lamp|lamps|chair|chairs|desk|desks|"
    r"bookshelf|bookshelves|cabinet|cabinets|bed|beds|plant|plants|"
    r"tree|trees|fence|fences|road segment|road segments|tv|tvs|book|books)"
)

def extract_quantity(text: str, default: int | None = None) -> int | None:
    s = text.lower()

    m = re.search(rf"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+{OBJECT_PATTERN}\b", s)
    if m:
        return NUMBER_WORDS[m.group(1)]

    m = re.search(rf"\b(\d+)\s+{OBJECT_PATTERN}\b", s)
    if m:
        return int(m.group(1))

    m = re.search(rf"\b(a|an|single)\s+{OBJECT_PATTERN}\b", s)
    if m:
        return 1

    return default


RELATION_TO_DISTANCE = {
    "near": 2.0,
    "on_top_of": 0.0,
    "inside": 0.0,
    "behind": 1.0,
    "in_front_of": 1.0,
    "next_to": 0.0,
    "between": 1.0,
    "under": 1.0,
    "none": 0.0,
}


# ========= Kategori helper'ları =========

OBJECT_CANON_MAP = {
    "sofa": "sofa", "sofas": "sofa",
    "table": "table", "tables": "table",
    "lamp": "lamp", "lamps": "lamp",
    "chair": "chair", "chairs": "chair",
    "desk": "desk", "desks": "desk",
    "bookshelf": "bookshelf", "bookshelves": "bookshelf",
    "cabinet": "cabinet", "cabinets": "cabinet",
    "bed": "bed", "beds": "bed",
    "plant": "plant", "plants": "plant",
    "tree": "tree", "trees": "tree",
    "fence": "fence", "fences": "fence",
    "road segment": "road_segment", "road segments": "road_segment",
    "tv": "tv", "tvs": "tv",
    "book": "book", "books": "book",
}

def find_category_in_phrase(phrase: str) -> str | None:
    s = phrase.lower()
    if "road segment" in s:
        return "road_segment"
    for w, c in OBJECT_CANON_MAP.items():
        if w in s:
            return c
    return None

def parse_behind_text_categories(text: str):
    s = text.lower()
    m = re.search(r"move\s+(?P<moved>.+?)\s+behind\s+the\s+(?P<ref>[^,\.]+)", s)
    if not m:
        return None, None
    moved_phrase = m.group("moved")
    ref_phrase   = m.group("ref")
    return find_category_in_phrase(moved_phrase), find_category_in_phrase(ref_phrase)

def category_from_span(span: str | None) -> str | None:
    if not span:
        return None
    return find_category_in_phrase(span)

def parse_ref_from_preposition(text: str, relation: str) -> str | None:
    s = text.lower()
    if relation == "in_front_of":
        m = re.search(r"in front of (the )?(?P<ref>[^,\.]+)", s)
    elif relation == "on_top_of":
        m = re.search(r"on top of (the )?(?P<ref>[^,\.]+)", s)
    else:
        return None

    if not m:
        return None

    ref_phrase = m.group("ref").strip()
    return find_category_in_phrase(ref_phrase)


# ========= RoleTagger span extraction =========

def extract_spans_with_roletagger(text, tokenizer, model, label_list, device):
    model.eval()
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = logits.argmax(dim=-1)[0].cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    labels = [label_list[p] for p in preds]

    def collect(tag):
        span = []
        active = False
        for tok, lab in zip(tokens, labels):
            if tok in ["[CLS]", "[SEP]"]:
                continue
            if lab == f"B-{tag}":
                if span:
                    break
                span = [tok]
                active = True
            elif lab == f"I-{tag}" and active:
                span.append(tok)
            else:
                if active:
                    break
        if not span:
            return None
        return tokenizer.convert_tokens_to_string(span).strip()

    return collect("MOVED"), collect("REF")


# ========= Model loader =========

def load_models_and_encoder():
    # -------- IntentHead --------
    intent_ckpt = torch.load(INTENT_MODEL_PATH, map_location="cpu")
    emb_dim = intent_ckpt["emb_dim"]
    intent_list = intent_ckpt["intent_list"]
    emb_model_name = intent_ckpt["embedding_model_name"]

    intent_model = IntentHead(emb_dim=emb_dim, num_intents=len(intent_list))
    intent_model.load_state_dict(intent_ckpt["state_dict"])
    intent_model.eval()

    # -------- SlotHead (v3) --------
    slot_ckpt = torch.load(SLOT_MODEL_PATH, map_location="cpu")

    if slot_ckpt["emb_dim"] != emb_dim:
        raise RuntimeError("Embedding dimension mismatch between IntentHead and SlotHead.")

    category_list = slot_ckpt["category_list"]
    relation_list = slot_ckpt["relation_list"]
    side_list = slot_ckpt["side_list"]
    quantity_list = slot_ckpt.get("quantity_list", [1])

    slot_model = SlotHeadV2(
        emb_dim=emb_dim,
        num_cat=len(category_list),
        num_rel=len(relation_list),
        num_side=len(side_list),
        num_qty=len(quantity_list),
    )
    slot_model.load_state_dict(slot_ckpt["state_dict"])
    slot_model.eval()

    # -------- RoleTagger (opsiyonel) --------
    role_tokenizer = None
    role_model = None
    role_label_list = None

    if ROLE_TAGGER_DIR.exists():
        print(f"Loading RoleTagger from folder: {ROLE_TAGGER_DIR}")
        role_tokenizer = BertTokenizerFast.from_pretrained(ROLE_TAGGER_DIR)
        role_model = BertForTokenClassification.from_pretrained(ROLE_TAGGER_DIR)
        role_model.eval()

        labels_path = ROLE_TAGGER_DIR / "labels.json"
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                labels_info = json.load(f)
            role_label_list = labels_info["label_list"]
        else:
            role_label_list = [role_model.config.id2label[i] for i in range(role_model.config.num_labels)]
    else:
        print("RoleTagger folder not found. Continuing WITHOUT it.")

    # -------- Relation modeli (override) --------
    relation_model = None
    if RELATION_MODEL_PATH.exists():
        try:
            raw = torch.load(RELATION_MODEL_PATH, map_location="cpu")

            if isinstance(raw, dict) and "state_dict" in raw:
                rel_state_dict = raw["state_dict"]
                rel_emb_dim = raw.get("emb_dim", emb_dim)
                rel_model_name = raw.get("embedding_model_name", emb_model_name)
                rel_list = raw.get("relation_list", relation_list)
                print("[RelationOverride] Detected metadata-style checkpoint.")
            elif isinstance(raw, dict):
                rel_state_dict = raw
                rel_emb_dim = emb_dim
                rel_model_name = emb_model_name
                rel_list = relation_list
                print("[RelationOverride] Detected plain state_dict checkpoint.")
            else:
                rel_state_dict = None
                rel_emb_dim = emb_dim
                rel_model_name = emb_model_name
                rel_list = relation_list
                print("[RelationOverride] Unknown checkpoint format. Override disabled.")

            if rel_state_dict is not None:
                if rel_emb_dim != emb_dim:
                    print("[RelationOverride] WARNING: emb_dim mismatch. Override disabled.")
                elif rel_model_name != emb_model_name:
                    print("[RelationOverride] WARNING: embedding model name mismatch. Override disabled.")
                elif list(rel_list) != list(relation_list):
                    print("[RelationOverride] WARNING: relation_list mismatch. Override disabled.")
                    print("  SlotHead relations :", relation_list)
                    print("  RelationHead list  :", rel_list)
                else:
                    relation_model = RelationHead(emb_dim=emb_dim, num_relations=len(relation_list))
                    relation_model.load_state_dict(rel_state_dict)
                    relation_model.eval()
                    print("[RelationOverride] Loaded relation_head and will override SlotHead relation_logits.")
        except Exception as e:
            print("[RelationOverride] ERROR while loading relation model:", e)
            relation_model = None
    else:
        print(f"[RelationOverride] relation model not found at {RELATION_MODEL_PATH}. Using SlotHead relations.")

    # -------- Object Attribute modeli (HF uyumlu) --------
    obj_attr_tokenizer = None
    obj_attr_model = None
    obj_attr_label_list = None

    if OBJECT_ATTR_DIR.exists():
        try:
            print(f"[ObjectAttr] Loading object attribute model from: {OBJECT_ATTR_DIR}")
            obj_attr_tokenizer = AutoTokenizer.from_pretrained(OBJECT_ATTR_DIR)
            obj_attr_model = AutoModelForSequenceClassification.from_pretrained(OBJECT_ATTR_DIR)
            obj_attr_model.eval()

            # HF config içinden label isimleri
            if hasattr(obj_attr_model.config, "id2label") and obj_attr_model.config.id2label:
                # id2label bazen dict: {"0":"x"} bazen {0:"x"} olabilir
                id2label = obj_attr_model.config.id2label
                # int index sırasına oturt
                obj_attr_label_list = [id2label[i] for i in range(obj_attr_model.config.num_labels)]
            else:
                obj_attr_label_list = list(ATTRS_FALLBACK)

            print(f"[ObjectAttr] OK loaded. num_labels={len(obj_attr_label_list)} labels={obj_attr_label_list}")

        except Exception as e:
            print("[ObjectAttr] ERROR while loading attribute model:", e)
            obj_attr_tokenizer = None
            obj_attr_model = None
            obj_attr_label_list = None
    else:
        print(f"[ObjectAttr] Folder not found at {OBJECT_ATTR_DIR}. Skipping attribute prediction.")

    # -------- SBERT encoder --------
    encoder = SentenceTransformer(emb_model_name)

    return (
        intent_model,
        slot_model,
        encoder,
        intent_list,
        category_list,
        relation_list,
        side_list,
        role_tokenizer,
        role_model,
        role_label_list,
        relation_model,
        obj_attr_tokenizer,
        obj_attr_model,
        obj_attr_label_list,
    )


# ========= Command builder =========

def build_command_json(
    intent: str,
    moved: str | None,
    ref: str | None,
    relation: str,
    side: str,
    qty: int,
    dist: float,
    moved_span: str | None,
    ref_span: str | None,
    model_full_id: str | None,
    object_attribute: list[str] | None,
):
    if intent in ["CreateObject", "PlaceObject", "MoveObject"]:
        args = {
            "category": moved,
            "reference_category": ref,
            "relation": relation,
            "side": side,
            "quantity": qty,
            "distance_m": dist,
            "object_attribute": object_attribute or [],
        }
    else:
        args = {}

    return {
        "intent": intent,
        "args": args,
        "model_full_id": model_full_id,
        "moved_span": moved_span,
        "reference_span": ref_span,
    }


# ========= Sentence → Command =========

def parse_sentence_to_command(
    sent,
    emb,
    intent_logits,
    slot_outputs,
    relation_override_logits,
    intent_list,
    category_list,
    relation_list,
    side_list,
    role_tokenizer,
    role_model,
    role_label_list,
    device,
    obj_attr_tokenizer,
    obj_attr_model,
    obj_attr_label_list,
):
    intent_id = int(intent_logits.argmax(dim=-1).item())
    intent = intent_list[intent_id]

    moved_idx = int(slot_outputs["moved_category_logits"].argmax(dim=-1).item())
    ref_idx   = int(slot_outputs["reference_category_logits"].argmax(dim=-1).item())

    if relation_override_logits is not None:
        rel_idx = int(relation_override_logits.argmax(dim=-1).item())
    else:
        rel_idx = int(slot_outputs["relation_logits"].argmax(dim=-1).item())

    side_idx  = int(slot_outputs["side_logits"].argmax(dim=-1).item())

    moved_raw = category_list[moved_idx] if 0 <= moved_idx < len(category_list) else None
    ref_raw   = category_list[ref_idx]   if 0 <= ref_idx   < len(category_list) else None
    relation  = relation_list[rel_idx]   if 0 <= rel_idx   < len(relation_list) else "none"
    side      = side_list[side_idx]      if 0 <= side_idx  < len(side_list)    else "none"

    low = sent.lower()

    has_left_of  = (" left of " in low) or (" to the left of " in low)
    has_right_of = (" right of " in low) or (" to the right of " in low)

    if intent in ["CreateObject", "PlaceObject", "MoveObject"]:
        if has_left_of or has_right_of:
            if relation in ["none", "near", "next_to"]:
                relation = "next_to"
            if has_left_of:
                side = "left"
            elif has_right_of:
                side = "right"

    qty = extract_quantity(sent, default=1)
    dist = RELATION_TO_DISTANCE.get(relation, 1.0)

    moved_final = moved_raw
    ref_final   = ref_raw
    relation_final = relation
    side_final     = side
    dist_final     = dist

    moved_span = None
    ref_span   = None
    if role_model is not None and role_tokenizer is not None and role_label_list is not None:
        moved_span, ref_span = extract_spans_with_roletagger(
            sent, role_tokenizer, role_model, role_label_list, device
        )

    if intent in ["CreateObject", "PlaceObject", "MoveObject"]:
        if relation_final == "behind":
            m_fix, r_fix = parse_behind_text_categories(sent)
            if m_fix and r_fix:
                moved_final = m_fix
                ref_final   = r_fix

        span_m_cat = category_from_span(moved_span)
        if span_m_cat is not None:
            moved_final = span_m_cat

        span_r_cat = category_from_span(ref_span)
        if span_r_cat is not None:
            ref_final = span_r_cat

        if relation_final in ["in_front_of", "on_top_of"]:
            parsed_ref = parse_ref_from_preposition(sent, relation_final)
            if parsed_ref is not None:
                ref_final = parsed_ref

        if intent == "CreateObject":
            has_spatial_kw = any(
                kw in low
                for kw in [
                    "near", "next to", "behind", "in front of",
                    "on top of", "between", "inside", "under",
                    "left of", "right of",
                ]
            )
            if not has_spatial_kw:
                relation_final = "none"
                side_final     = "none"
                ref_final      = None
                dist_final     = 0.0
    else:
        relation_final = "none"
        side_final     = "none"
        dist_final     = 0.0

    model_full_id = None
    object_attribute = []

    if intent in ["CreateObject", "PlaceObject", "MoveObject"]:
        lookup_key = moved_span if moved_span else moved_final
        model_full_id = pick_shapenet_model(lookup_key)

        # --- ObjectAttr: moved nesnesine göre attribute tahmini ---
        attr_key = (moved_span or moved_final or "").strip()
        object_attribute = predict_object_attributes(
            obj_text=attr_key,
            tokenizer=obj_attr_tokenizer,
            model=obj_attr_model,
            label_list=obj_attr_label_list or [],
            device=device,
            threshold=0.50,
            max_len=64,
        )

    cmd = build_command_json(
        intent=intent,
        moved=moved_final,
        ref=ref_final,
        relation=relation_final,
        side=side_final,
        qty=qty,
        dist=dist_final,
        moved_span=moved_span,
        ref_span=ref_span,
        model_full_id=model_full_id,
        object_attribute=object_attribute,
    )

    print("\n=== Parsed ===")
    print("Sentence:", sent)
    print("Intent:", intent)
    print("Relation:", relation_final, "| Side:", side_final)
    print("Moved category:", moved_final, "| Ref category:", ref_final)
    print("Quantity:", qty, "| Distance_m:", round(dist_final, 2))
    print("Moved span:", moved_span)
    print("Reference span:", ref_span)
    print("Model full_id:", model_full_id)
    print("Object attribute (>=0.50):", object_attribute)
    print("Command JSON:")
    print(json.dumps(cmd, indent=2, ensure_ascii=False))

    return cmd


# ========= write file =========

def write_next_command(cmd):
    NEXT_COMMAND_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NEXT_COMMAND_PATH.open("w", encoding="utf-8") as f:
        json.dump(cmd, f, indent=2, ensure_ascii=False)
    print(f"[OK] JSON yazıldı → {NEXT_COMMAND_PATH}")


# ========= Interactive loop =========

def interactive_loop(
    intent_model,
    slot_model,
    encoder,
    intent_list,
    category_list,
    relation_list,
    side_list,
    role_tokenizer,
    role_model,
    role_label_list,
    relation_model,
    obj_attr_tokenizer,
    obj_attr_model,
    obj_attr_label_list,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    intent_model.to(device)
    slot_model.to(device)
    if role_model is not None:
        role_model.to(device)
    if relation_model is not None:
        relation_model.to(device)
    if obj_attr_model is not None:
        obj_attr_model.to(device)

    print("\nHazır. Komut gir:")
    print(f"(JSON çıkışı: {NEXT_COMMAND_PATH})")
    print("(q / quit / exit ile çıkılır)\n")

    while True:
        try:
            text = input(">>> ").strip()
        except EOFError:
            print("\nEOF – çıkılıyor.")
            break

        if not text:
            continue

        low = text.lower()
        if low in {"q", "quit", "exit"}:
            print("Çıkılıyor...")
            break

        if text.lstrip().startswith("{"):
            try:
                cmd = json.loads(text)
            except Exception as e:
                print("JSON parse hatası:", e)
                continue
            write_next_command(cmd)
            continue

        with torch.no_grad():
            emb = encoder.encode([text], convert_to_tensor=True, show_progress_bar=False).to(device)

            intent_logits = intent_model(emb)
            slot_outputs  = slot_model(emb)

            if relation_model is not None:
                relation_override_logits = relation_model(emb)
            else:
                relation_override_logits = None

        cmd = parse_sentence_to_command(
            text,
            emb,
            intent_logits,
            slot_outputs,
            relation_override_logits,
            intent_list,
            category_list,
            relation_list,
            side_list,
            role_tokenizer,
            role_model,
            role_label_list,
            device,
            obj_attr_tokenizer,
            obj_attr_model,
            obj_attr_label_list,
        )

        write_next_command(cmd)


# ========= main =========

def main():
    (
        intent_model,
        slot_model,
        encoder,
        intent_list,
        category_list,
        relation_list,
        side_list,
        role_tokenizer,
        role_model,
        role_label_list,
        relation_model,
        obj_attr_tokenizer,
        obj_attr_model,
        obj_attr_label_list,
    ) = load_models_and_encoder()

    interactive_loop(
        intent_model,
        slot_model,
        encoder,
        intent_list,
        category_list,
        relation_list,
        side_list,
        role_tokenizer,
        role_model,
        role_label_list,
        relation_model,
        obj_attr_tokenizer,
        obj_attr_model,
        obj_attr_label_list,
    )

if __name__ == "__main__":
    main()
