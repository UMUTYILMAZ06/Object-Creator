# -*- coding: utf-8 -*-
# run_command_loop.py
#
# Purpose:
#  - Loads the models: intent_head_v2.pt + slot_head_v3.pt + (optional) RoleTagger (HF folder)
#  - Takes an English sentence from the console
#  - Produces a command JSON using intent + slot + (if available) RoleTagger + small heuristics
#  - Writes the JSON to the next_command.json file that Unity reads
#  - For moved_category, selects a model fullId from metadata.csv and adds it to the JSON



import re
import json
import csv
import random
import argparse
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



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json-out", type=str, default=None, help="Full path to next_command.json")
    return p.parse_args()

ARGS = parse_args()


# ======= Paths =======

INTENT_MODEL_PATH   = Path("intent_head_v2.pt")
SLOT_MODEL_PATH     = Path("slot_head_v3.pt")

# RoleTagger (for spans, HF folder)
ROLE_TAGGER_DIR     = Path("roletagger_bert_v1")

# # Relation model (SBERT + MLP)
# RELATION_MODEL_PATH = Path("relation_head_v1.pt")

# Relation token model folder
RELATION_TOKEN_DIR = Path("relationtoken")  # ai_engine/relationtoken
REL_CUE_MAPPING_PATH = RELATION_TOKEN_DIR / "cue_mapping.json"
REL_LABELS_PATH      = RELATION_TOKEN_DIR / "labels.json"

# ObjectType HF model folder    
OBJECTTYPE_DIR = Path("ObjectType")  # ai_engine/ObjectType

# Path to ShapeNet metadata.csv
METADATA_CSV_PATH = Path(r"D:\ShapeNetSem\ShapeNetSem-backup\metadata.csv")

# Output command JSON path (read by Unity)
NEXT_COMMAND_PATH = Path(
    r"C:\Users\Eren\Desktop\Aktif Projeler\Mergen\Assets\StreamingAssets\next_command.json"
)
if ARGS.json_out:
    NEXT_COMMAND_PATH = Path(ARGS.json_out)
    print(f"[CLI] Overriding NEXT_COMMAND_PATH -> {NEXT_COMMAND_PATH}")

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
# class RelationHead(nn.Module):
#     """
#     SBERT embedding'leri üzerinde relation sınıflandırma yapan MLP.
#     SlotHead'in relation_logits çıktısını override etmek için kullanılır.
#     """
#     def __init__(self, emb_dim: int, num_relations: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(emb_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, num_relations),
#         )

#     def forward(self, x):
#         return self.net(x)


# ========= SlotHead (v3 compatible) =========

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


# ========= Quantity & distance heuristics =========

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

OBJECT_PATTERN = (
    r"(?:sofa|sofas|table|tables|lamp|lamps|chair|chairs|desk|desks|"
    r"bookshelf|bookshelves|cabinet|cabinets|bed|beds|plant|plants|"
    r"tree|trees|fence|fences|road segment|road segments|tv|tvs|book|books|"
    r"apple|apples)"
)

KNOWN_MATERIALS = {
    "wood","metal","plastic","glass","stone","marble","concrete","fabric","leather",
    "gold","silver","copper","chrome"
}

def extract_quantity(text: str, default: int | None = None) -> int | None:
    s = text.lower()

    m = re.search(
        rf"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+{OBJECT_PATTERN}\b",
        s,
    )
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

# ========= Extra arg extractors (Rotate/Resize/Room/Material) =========

def extract_rotate_degrees(text: str, default: float = 45.0) -> float:
    """
    Returns degrees. Negative = counterclockwise/left, Positive = clockwise/right.
    Examples:
      "rotate the chair 45 degrees" -> 45
      "rotate chair left" -> -45
    """
    s = text.lower()
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:degrees?|°)\b", s)
    if m:
        return float(m.group(1))

    # fallback: any number after rotate
    m = re.search(r"\brotate\b[^0-9-]*(-?\d+(?:\.\d+)?)\b", s)
    if m:
        return float(m.group(1))

    if any(k in s for k in ["left", "counterclockwise", "anticlockwise"]):
        return -abs(default)
    if any(k in s for k in ["right", "clockwise"]):
        return abs(default)

    return default


def extract_scale_factor(text: str, default: float = 1.25) -> float:
    """
    Returns multiplicative scale factor.
    Examples:
      "resize the lamp 2x" -> 2.0
      "make it smaller" -> 0.8
    """
    s = text.lower()

    m = re.search(r"\b(\d+(?:\.\d+)?)\s*x\b", s)   # 2x, 1.5x
    if m:
        return float(m.group(1))

    m = re.search(r"\b(\d+(?:\.\d+)?)\s*%\b", s)   # 20%
    if m:
        p = float(m.group(1)) / 100.0
        if any(k in s for k in ["decrease", "smaller", "shrink", "reduce", "down"]):
            return max(0.01, 1.0 - p)
        return 1.0 + p

    if any(k in s for k in ["smaller", "shrink", "reduce", "decrease", "down"]):
        return 0.8
    if any(k in s for k in ["bigger", "larger", "increase", "grow", "up"]):
        return 1.25

    return default


def extract_room_dims(text: str, default=(6.0, 8.0, 3.0)):
    """
    Parses '6x8x3' style room dimensions (width x length x height).
    """
    s = text.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)", s)
    if m:
        return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    return default


def extract_material_name(text: str) -> str | None:
    s = text.lower()

    # "to wood"
    m = re.search(r"\bto\s+([a-z_]+)\b", s)
    if m:
        cand = m.group(1)
        return cand

    # known materials
    for mat in KNOWN_MATERIALS:
        if re.search(rf"\b{re.escape(mat)}\b", s):
            return mat

    return None
# ========= Category helpers =========

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
    "apple": "apple", "apples": "apple",
}

CATEGORY_ALIASES = {
    "sectional": "sofa",
    "couch": "sofa",
    "settee": "sofa",
}

def normalize_category(cat: str | None) -> str | None:
    if not cat:
        return cat
    c = cat.strip().lower()
    return CATEGORY_ALIASES.get(c, c)

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


# ===================== Relation via relationtoken =====================

def _normalize_cue(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def load_relationtoken():
    if not RELATION_TOKEN_DIR.exists():
        raise FileNotFoundError(f"[RelationToken] Folder not found: {RELATION_TOKEN_DIR}")

    tok = BertTokenizerFast.from_pretrained(RELATION_TOKEN_DIR)
    mdl = BertForTokenClassification.from_pretrained(RELATION_TOKEN_DIR)
    mdl.eval()

    if REL_LABELS_PATH.exists():
        info = json.loads(REL_LABELS_PATH.read_text(encoding="utf-8"))
        label_list = info.get("label_list")
    else:
        label_list = [mdl.config.id2label[i] for i in range(mdl.config.num_labels)]

    cue_map = {}
    if REL_CUE_MAPPING_PATH.exists():
        cue_map_raw = json.loads(REL_CUE_MAPPING_PATH.read_text(encoding="utf-8"))
        cue_map = {_normalize_cue(k): v for k, v in cue_map_raw.items()}

    # fallback defaults
    defaults = {
        "behind": "behind",
        "in front of": "in_front_of",
        "on top of": "on_top_of",
        "on": "on_top_of",
        "inside": "inside",
        "in": "inside",
        "between": "between",
        "next to": "next_to",
        "near": "near",
        "under": "under",
        "below": "under",
        "left of": "next_to",
        "right of": "next_to",
        "close to": "near",
        "beside": "next_to",
        "underneath": "under",
    }
    for k, v in defaults.items():
        cue_map.setdefault(_normalize_cue(k), v)

    print(f"[RelationToken] Loaded from {RELATION_TOKEN_DIR}")
    return tok, mdl, label_list, cue_map

def predict_relation_from_relationtoken(text: str, tok, mdl, label_list, cue_map: dict, device):
    """
    Returns: (relation_label, cue_span_or_None)
    """
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = mdl(**enc)
        pred = out.logits.argmax(dim=-1)[0].cpu().tolist()

    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0].cpu())
    labels = [label_list[i] for i in pred]

    cue_tokens = []
    active = False
    for t, lab in zip(tokens, labels):
        if t in ["[CLS]", "[SEP]"]:
            continue
        if lab == "B-CUE":
            cue_tokens = [t]
            active = True
        elif lab == "I-CUE" and active:
            cue_tokens.append(t)
        else:
            if active:
                break

    cue_span = tok.convert_tokens_to_string(cue_tokens).strip() if cue_tokens else None
    if cue_span:
        key = _normalize_cue(cue_span)
        rel = cue_map.get(key)
        if rel:
            return rel, cue_span

    return "none", cue_span

# ===================== ObjectType (HF multi-label) =====================

def load_objecttype():
    """
    Expects HF folder:
      ai_engine/ObjectType/
        config.json
        model.safetensors
        tokenizer.json ...
    And optionally labels file:
      ai_engine/ObjectType/labels.json  -> {"label_list":[...]}
      OR ai_engine/ObjectType/multilabel_meta.json -> {"label_list":[...]}
    """
    if not OBJECTTYPE_DIR.exists():
        print(f"[ObjectType] WARNING: Folder not found: {OBJECTTYPE_DIR} (skip)")
        return None, None, None

    tok = AutoTokenizer.from_pretrained(OBJECTTYPE_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(OBJECTTYPE_DIR)
    mdl.eval()

    label_list = None
    labels_json = OBJECTTYPE_DIR / "labels.json"
    meta_json   = OBJECTTYPE_DIR / "multilabel_meta.json"
    if labels_json.exists():
        info = json.loads(labels_json.read_text(encoding="utf-8"))
        label_list = info.get("label_list")
    elif meta_json.exists():
        info = json.loads(meta_json.read_text(encoding="utf-8"))
        label_list = info.get("label_list")

    if not label_list:
        # fallback: id2label
        try:
            label_list = [mdl.config.id2label[i] for i in range(mdl.config.num_labels)]
        except Exception:
            label_list = None

    print(f"[ObjectType] Loaded from {OBJECTTYPE_DIR}")
    return tok, mdl, label_list

def predict_objecttype_topk(text: str, tok, mdl, label_list, device, threshold: float = 0.50, topk: int = 3):
    """
    Multi-label: sigmoid(logits) -> pick >= threshold -> sort desc -> take topk
    Returns: list[str]
    """
    if not text or tok is None or mdl is None or not label_list:
        return []

    enc = tok(text, return_tensors="pt", truncation=True, max_length=64)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = mdl(**enc)
        logits = out.logits  # [1, num_labels]
        probs = torch.sigmoid(logits)[0].detach().cpu().tolist()

    scored = []
    for i, p in enumerate(probs):
        if i < len(label_list) and p >= threshold:
            scored.append((p, label_list[i]))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [lab for _, lab in scored[:topk]]

# ========= Model loader =========

# def load_relation_checkpoint_mandatory():
#     """relation_head_v1.pt zorunlu. Checkpoint metadata-style olmalı:
#     {state_dict, relation_list, embedding_model_name, emb_dim}
#     """
#     if not RELATION_MODEL_PATH.exists():
#         raise FileNotFoundError(
#             f"[Relation] ZORUNLU dosya yok: {RELATION_MODEL_PATH}\n"
#             f"-> relation_head_v1.pt olmadan relation tahmini yapılamaz."
#         )

#     raw = torch.load(RELATION_MODEL_PATH, map_location="cpu")

#     if isinstance(raw, dict) and "state_dict" in raw:
#         rel_state_dict = raw["state_dict"]
#         rel_emb_dim = int(raw.get("emb_dim")) if raw.get("emb_dim") is not None else None
#         rel_model_name = raw.get("embedding_model_name")
#         rel_list = raw.get("relation_list")
#         if not rel_model_name or not rel_list or not rel_emb_dim:
#             raise RuntimeError("[Relation] Checkpoint içinde embedding_model_name / relation_list / emb_dim eksik.")
#     elif isinstance(raw, dict):
#         # plain state_dict: metadata yok -> bu sürümde kabul etmiyoruz
#         raise RuntimeError(
#             "[Relation] relation_head_v1.pt plain state_dict gibi görünüyor.\n"
#             "-> Lütfen checkpoint'i metadata ile kaydet: {state_dict, relation_list, embedding_model_name, emb_dim}"
#         )
#     else:
#         raise RuntimeError("[Relation] relation_head_v1.pt formatı tanınmadı.")

#     print(f"[Relation] Loaded checkpoint from {RELATION_MODEL_PATH}")
#     print(f"[Relation] embedding_model_name={rel_model_name} emb_dim={rel_emb_dim} rels={list(rel_list)}")

#     return rel_state_dict, rel_emb_dim, rel_model_name, list(rel_list)


def load_models_and_encoder():
    # Intent
    intent_ckpt = torch.load(INTENT_MODEL_PATH, map_location="cpu")
    emb_dim = intent_ckpt["emb_dim"]
    intent_list = intent_ckpt["intent_list"]
    emb_model_name = intent_ckpt["embedding_model_name"]

    intent_model = IntentHead(emb_dim=emb_dim, num_intents=len(intent_list))
    intent_model.load_state_dict(intent_ckpt["state_dict"])
    intent_model.eval()

    # Slot
    slot_ckpt = torch.load(SLOT_MODEL_PATH, map_location="cpu")
    if slot_ckpt["emb_dim"] != emb_dim:
        raise RuntimeError("Embedding dimension mismatch between IntentHead and SlotHead.")

    category_list = slot_ckpt["category_list"]
    relation_list = slot_ckpt["relation_list"]  # fallback only
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

    # RoleTagger
    role_tokenizer = None
    role_model = None
    role_label_list = None
    if ROLE_TAGGER_DIR.exists():
        role_tokenizer = BertTokenizerFast.from_pretrained(ROLE_TAGGER_DIR)
        role_model = BertForTokenClassification.from_pretrained(ROLE_TAGGER_DIR)
        role_model.eval()

        labels_path = ROLE_TAGGER_DIR / "labels.json"
        if labels_path.exists():
            info = json.loads(labels_path.read_text(encoding="utf-8"))
            role_label_list = info["label_list"]
        else:
            role_label_list = [role_model.config.id2label[i] for i in range(role_model.config.num_labels)]

    # Encoder
    encoder_main = SentenceTransformer(emb_model_name)

    # RelationToken
    rel_tok, rel_mdl, rel_label_list, cue_map = load_relationtoken()

    # ObjectType
    obj_tok, obj_mdl, obj_label_list = load_objecttype()

    return (
        intent_model,
        slot_model,
        encoder_main,
        intent_list,
        category_list,
        relation_list,
        side_list,
        role_tokenizer,
        role_model,
        role_label_list,
        rel_tok,
        rel_mdl,
        rel_label_list,
        cue_map,
        obj_tok,
        obj_mdl,
        obj_label_list,
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
    object_attribute: list[str] | None = None, 
    cue_span: str | None = None,               
    extra_args: dict | None = None,
):
    args = {}

    if intent in ["CreateObject", "PlaceObject", "MoveObject"]:
        args.update({
            "category": moved,
            "reference_category": ref,
            "relation": relation,
            "side": side,
            "quantity": qty,
            "distance_m": dist,
        })
        if object_attribute is not None:
            args["object_attribute"] = object_attribute
        if cue_span is not None:
            args["cue_span"] = cue_span  

    if extra_args:
        args.update(extra_args)

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
    # relationtoken:
    rel_tok,
    rel_mdl,
    rel_label_list,
    cue_map,
    # objecttype:
    obj_tok,
    obj_mdl,
    obj_label_list,
    device,
):
    intent_id = int(intent_logits.argmax(dim=-1).item())
    intent = intent_list[intent_id]

    moved_idx = int(slot_outputs["moved_category_logits"].argmax(dim=-1).item())
    ref_idx   = int(slot_outputs["reference_category_logits"].argmax(dim=-1).item())
    side_idx  = int(slot_outputs["side_logits"].argmax(dim=-1).item())

    moved_raw = category_list[moved_idx] if 0 <= moved_idx < len(category_list) else None
    ref_raw   = category_list[ref_idx]   if 0 <= ref_idx   < len(category_list) else None
    side      = side_list[side_idx]      if 0 <= side_idx  < len(side_list)    else "none"

    low = sent.lower()

    # --- Relation: primary = relationtoken ---
    relation_primary, cue_span = predict_relation_from_relationtoken(
        sent, rel_tok, rel_mdl, rel_label_list, cue_map, device
    )
    if not relation_primary:
        relation_primary = "none"

    # --- old ---
    if relation_override_logits is not None:
        rel_idx = int(relation_override_logits.argmax(dim=-1).item())
        relation_head = relation_list[rel_idx] if 0 <= rel_idx < len(relation_list) else "none"
        relation_primary = relation_head  # override

    # --- left/right heuristics -> next_to + side=left/right ---
    has_left_of  = (" left of " in low) or (" to the left of " in low)
    has_right_of = (" right of " in low) or (" to the right of " in low)
    if intent in ["CreateObject", "PlaceObject", "MoveObject"]:
        if has_left_of or has_right_of:
            if relation_primary in ["none", "near", "next_to"]:
                relation_primary = "next_to"
            side = "left" if has_left_of else "right"

    qty = extract_quantity(sent, default=1) or 1
    dist = RELATION_TO_DISTANCE.get(relation_primary, 1.0)

    moved_final = moved_raw
    ref_final   = ref_raw
    relation_final = relation_primary
    side_final     = side
    dist_final     = dist

    # --- RoleTagger span extraction (optional) ---
    moved_span = None
    ref_span   = None
    if role_model is not None and role_tokenizer is not None and role_label_list is not None:
        moved_span, ref_span = extract_spans_with_roletagger(
            sent, role_tokenizer, role_model, role_label_list, device
        )

    if intent in ["CreateObject", "PlaceObject", "MoveObject"]:
        # special "behind" textual parsing
        if relation_final == "behind":
            m_fix, r_fix = parse_behind_text_categories(sent)
            if m_fix and r_fix:
                moved_final = m_fix
                ref_final   = r_fix

        # span -> category override
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
            has_phrase_kw = any(
                kw in low for kw in [
                    "near", "next to", "behind", "in front of", "on top of",
                    "between", "inside", "under", "left of", "right of",
                    "close to", "beside", "underneath",
                ]
            )
            has_on = re.search(r"\bon\b", low) is not None
            has_in = re.search(r"\bin\b", low) is not None
            has_spatial_kw = has_phrase_kw or has_on or has_in

            if not has_spatial_kw:
                relation_final = "none"
                side_final     = "none"
                ref_final      = None
                dist_final     = 0.0

    else:
        # non-placement intents should not carry spatial relation
        relation_final = "none"
        side_final     = "none"
        dist_final     = 0.0

        # Delete special: "delete all <obj>"
        if intent == "DeleteObject":
            m_all = re.search(r"\b(delete|remove)\s+all\s+(?P<obj>[^\.]+)", low)
            if m_all:
                obj_phrase = m_all.group("obj").strip()
                cat = find_category_in_phrase(obj_phrase)
                if cat is not None:
                    moved_final = cat
                if moved_span is None and ref_span is None:
                    moved_span = obj_phrase

        # span fallback
        if moved_span is None and ref_span is None:
            if moved_raw is not None:
                moved_span = moved_raw
            elif ref_raw is not None:
                ref_span = ref_raw

    # --- ShapeNet model selection (only for create/place/move) ---
    model_full_id = None
    if intent in ["CreateObject", "PlaceObject", "MoveObject"]:
        lookup_key = moved_span if moved_span else moved_final
        model_full_id = pick_shapenet_model(lookup_key)

    # --- ObjectType(topk) ---
    object_attribute = None
    if obj_tok is not None and obj_mdl is not None and obj_label_list is not None:
        obj_query = moved_span if moved_span else (moved_final or "")
        object_attribute = predict_objecttype_topk(
            obj_query,
            obj_tok,
            obj_mdl,
            obj_label_list,
            device=device,
            threshold=0.50,
            topk=3,
        )

    # --- EXTRA ARGS for Unity ---
    extra_args = {}

    if intent == "RotateObject":
        extra_args["category"] = moved_final
        extra_args["rotate_degrees"] = extract_rotate_degrees(sent)

    elif intent == "ResizeObject":
        extra_args["category"] = moved_final
        extra_args["scale_factor"] = extract_scale_factor(sent)

    elif intent == "SetRoom":
        w, l, h = extract_room_dims(sent)
        extra_args["room_width"] = w
        extra_args["room_length"] = l
        extra_args["room_height"] = h

    elif intent == "DeleteObject":
        delete_all = bool(re.search(r"\b(delete|remove)\s+all\b", low))
        extra_args["category"] = moved_final
        extra_args["quantity"] = 0 if delete_all else max(1, qty)

    elif intent == "SetMaterial":
        if (not moved_span) and ref_span:
            moved_span = ref_span
            ref_span = None

        moved_final = normalize_category(moved_final)

        mat = extract_material_name(sent)
        extra_args["category"] = moved_final
        if mat:
            extra_args["material_name"] = mat

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
        cue_span=cue_span,                      
        extra_args=extra_args if extra_args else None,
    )

    print("\n=== Parsed ===")
    print("Sentence:", sent)
    print("Intent:", intent)
    print("Relation:", relation_final, "| Side:", side_final, "| Cue:", cue_span)
    print("Moved category:", moved_final, "| Ref category:", ref_final)
    print("Quantity:", qty, "| Distance_m:", round(dist_final, 2))
    print("Moved span:", moved_span)
    print("Reference span:", ref_span)
    print("Model full_id:", model_full_id)
    print("ObjectType(top3@0.50):", object_attribute)
    print("Command JSON:")
    print(json.dumps(cmd, indent=2))

    return cmd


# ========= write file =========

def write_next_command(cmd):
    NEXT_COMMAND_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NEXT_COMMAND_PATH.open("w", encoding="utf-8") as f:
        json.dump(cmd, f, indent=2, ensure_ascii=False)
    print(f"[OK] JSON written-> {NEXT_COMMAND_PATH}")


# ========= Interactive loop =========

def interactive_loop(
    intent_model,
    slot_model,
    encoder_main,
    intent_list,
    category_list,
    relation_list_fallback,
    side_list,
    role_tokenizer,
    role_model,
    role_label_list,
    rel_tok,
    rel_mdl,
    rel_label_list,
    cue_map,
    obj_tok,
    obj_mdl,
    obj_label_list,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    intent_model.to(device)
    slot_model.to(device)
    if role_model is not None:
        role_model.to(device)
    rel_mdl.to(device)
    if obj_mdl is not None:
        obj_mdl.to(device)

    print("[READY]")
    print(f"JSON output: {NEXT_COMMAND_PATH}")
    print("(Exit is done with q / quit / exit)\n")

    while True:
        try:
            text = input(">>> ").strip()
        except EOFError:
            break

        if not text:
            continue
        if text.lower() in {"q", "quit", "exit"}:
            break

        with torch.no_grad():
            emb_main = encoder_main.encode([text], convert_to_tensor=True, show_progress_bar=False).to(device)
            intent_logits = intent_model(emb_main)
            slot_outputs  = slot_model(emb_main)

        cmd = parse_sentence_to_command(
                text,
                emb_main,
                intent_logits,
                slot_outputs,
                None,                     
                intent_list,
                category_list,
                relation_list_fallback,
                side_list,
                role_tokenizer,
                role_model,
                role_label_list,
                rel_tok,
                rel_mdl,
                rel_label_list,
                cue_map,
                obj_tok,
                obj_mdl,
                obj_label_list,
                device,
            )



        write_next_command(cmd)

def main():
    (
        intent_model,
        slot_model,
        encoder_main,
        intent_list,
        category_list,
        relation_list_fallback,
        side_list,
        role_tokenizer,
        role_model,
        role_label_list,
        rel_tok,
        rel_mdl,
        rel_label_list,
        cue_map,
        obj_tok,
        obj_mdl,
        obj_label_list,
    ) = load_models_and_encoder()

    interactive_loop(
        intent_model,
        slot_model,
        encoder_main,
        intent_list,
        category_list,
        relation_list_fallback,
        side_list,
        role_tokenizer,
        role_model,
        role_label_list,
        rel_tok,
        rel_mdl,
        rel_label_list,
        cue_map,
        obj_tok,
        obj_mdl,
        obj_label_list,
    )

if __name__ == "__main__":
    main()

