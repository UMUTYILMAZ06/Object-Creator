# train_intent_v1.py

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer


# ==========
#  Config
# ==========

DATA_PATH = Path("sentences_v1.jsonl")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SAVE_PATH = Path("intent_head_v1.pt")

# Intent listesi – Unity tarafında da aynı sırayı kullan
INTENTS = [
    "SetRoom",
    "CreateObject",
    "PlaceObject",
    "RotateObject",
    "ResizeObject",
    "DeleteObject",
    "SetMaterial",
]

INTENT2ID = {name: i for i, name in enumerate(INTENTS)}
NUM_INTENTS = len(INTENTS)

BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 42


# ==========
#  Utils
# ==========

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(obj)
    return data


# ==========
#  Dataset
# ==========

class IntentDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        embeddings: [N, emb_dim]
        labels: [N]
        """
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ==========
#  Model
# ==========

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


# ==========
#  Train / Eval
# ==========

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        batch_size = yb.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            batch_size = yb.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


# ==========
#  Main
# ==========

def main():
    set_seed(SEED)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print(f"Loading data from {DATA_PATH} ...")
    raw_data = load_jsonl(DATA_PATH)

    texts = []
    labels = []

    for item in raw_data:
        text = item["text"]
        intent_name = item["gold_command"]["intent"]
        if intent_name not in INTENT2ID:
            # Eğer INTENTS listesinde olmayan intent varsa, şimdilik atla
            continue
        texts.append(text)
        labels.append(INTENT2ID[intent_name])

    labels = torch.tensor(labels, dtype=torch.long)
    print(f"Total usable samples: {len(texts)}")

    # ---- SBERT encoder (freeze) ----
    print(f"Loading SBERT encoder: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME)

    print("Encoding sentences with SBERT (this may take a bit)...")
    # Tek seferde embedding üret, sonra MLP için kullan
    embeddings = encoder.encode(
        texts,
        convert_to_tensor=True,
        batch_size=32,
        show_progress_bar=True
    )
    emb_dim = embeddings.size(1)
    print(f"Embedding shape: {embeddings.shape}")

    # ---- Train / Val split ----
    indices = list(range(len(texts)))
    random.shuffle(indices)

    split = int(len(indices) * (1 - VAL_SPLIT))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_emb = embeddings[train_idx]
    train_labels = labels[train_idx]

    val_emb = embeddings[val_idx]
    val_labels = labels[val_idx]

    train_ds = IntentDataset(train_emb, train_labels)
    val_ds = IntentDataset(val_emb, val_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Model / Optim / Loss ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = IntentHead(emb_dim=emb_dim, num_intents=NUM_INTENTS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- Training loop ----
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        print(
            f"[Epoch {epoch:02d}/{EPOCHS}] "
            f"TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} "
            f"TrainAcc={train_acc:.3f} ValAcc={val_acc:.3f}"
        )

        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_obj = {
                "state_dict": model.state_dict(),
                "intent_list": INTENTS,
                "embedding_model_name": MODEL_NAME,
                "emb_dim": emb_dim,
            }
            torch.save(save_obj, SAVE_PATH)
            print(f"  -> New best model saved to {SAVE_PATH} (ValAcc={val_acc:.3f})")

    print("Training finished.")


if __name__ == "__main__":
    main()
