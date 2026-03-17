import csv
from pathlib import Path

SPLIT_ROOT = Path("/scratch/ali95/kvasir-dataset/kvasir_split_70_20_10")
SPLITS = ["train", "val", "test"]

CLASSES = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def list_items(folder: Path):
    if not folder.exists():
        return []
    items = []
    for p in folder.iterdir():  # split/class/*
        if p.is_file() or p.is_symlink():
            items.append(p)
    items.sort(key=lambda x: x.name)
    return items

print("[INFO] SPLIT_ROOT:", SPLIT_ROOT, "exists=", SPLIT_ROOT.exists())

for split in SPLITS:
    out_csv = SPLIT_ROOT / f"{split}.csv"
    rows = []

    split_dir = SPLIT_ROOT / split
    print(f"\n[INFO] split={split} dir={split_dir} exists={split_dir.exists()}")

    for cls in CLASSES:
        cls_dir = split_dir / cls
        items = list_items(cls_dir)
        label_idx = CLASS_TO_IDX[cls]

        print(f"  [INFO] class={cls} count={len(items)} dir={cls_dir}")

        for p in items:
            image_id = p.stem  # filename without extension
            rows.append((image_id, str(p), cls, label_idx))

    rows.sort(key=lambda x: x[1])  # sort by path

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "path", "label", "label_idx"])
        w.writerows(rows)

    print(f"[OK] wrote {out_csv} rows={len(rows)}")

print("\nExpected totals: train=2800, val=800, test=400")
