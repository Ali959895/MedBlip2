#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize HAM10000 predictions (correct vs incorrect) WITHOUT touching core code.

What it does:
  1) Loads config + dataloaders from src/run.py (CsvImageDataset includes "path").
  2) Loads a checkpoint (trainable-only .pt) using load_trainable_checkpoint from src/run.py.
  3) Runs inference on val or test split.
  4) Writes:
      - predictions.csv
      - confusion_matrix.png
      - gallery.html (Correct section + Incorrect section)

Example:
  python scripts/visualize_predictions.py -c configs/ham10000_finetune.yaml \
      --ckpt /path/to/run_dir/best_trainable.pt --split test --max_items 200

Notes:
  - Uses argmax of softmax probabilities for predicted class.
  - "Corrected" = correct prediction (pred == true)
  - "Uncorrected" = incorrect prediction
"""
import argparse
import base64
import csv
import io
import json
import os
import sys
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Ensure imports work WITHOUT changing core code.
#
# In this repo, core modules are imported as top-level packages like `vlm.*`.
# That requires adding the `src/` directory (not just repo root) to PYTHONPATH.
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(PROJ, "src")
for p in (SRC, PROJ):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import core helpers from src/run.py as a module (no `src.` package required).
import run as run_mod  # noqa: E402

load_yaml = run_mod.load_yaml
build_ham_dataloaders = run_mod.build_ham_dataloaders
build_model = run_mod.build_model
load_trainable_checkpoint = run_mod.load_trainable_checkpoint


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


@torch.no_grad()
def run_inference(model, loader, device, classes: List[str], max_items: int = 0):
    model.eval()
    rows = []
    y_true_all = []
    y_pred_all = []

    n = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        paths = batch.get("path", None)
        if paths is None:
            # fallback: create dummy paths
            paths = [""] * x.shape[0]

        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)

        for i in range(x.shape[0]):
            yt = int(y[i].item())
            yp = int(pred[i].item())
            c = float(conf[i].item())
            p = str(paths[i])
            rows.append(
                {
                    "path": p,
                    "true_idx": yt,
                    "true": classes[yt] if yt < len(classes) else str(yt),
                    "pred_idx": yp,
                    "pred": classes[yp] if yp < len(classes) else str(yp),
                    "confidence": c,
                    "correct": int(yt == yp),
                }
            )
            y_true_all.append(yt)
            y_pred_all.append(yp)

            n += 1
            if max_items and n >= max_items:
                return rows, np.array(y_true_all), np.array(y_pred_all)

    return rows, np.array(y_true_all), np.array(y_pred_all)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], out_png: str):
    # confusion matrix (no sklearn dependency)
    k = len(classes)
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < k and 0 <= p < k:
            cm[t, p] += 1

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    # annotate a few cells if small enough
    if k <= 15:
        for i in range(k):
            for j in range(k):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _img_to_data_uri(path: str, max_side: int = 256) -> str:
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = min(1.0, float(max_side) / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


def write_gallery(rows: List[Dict[str, Any]], out_html: str, title: str, max_side: int = 256):
    correct = [r for r in rows if int(r["correct"]) == 1]
    wrong = [r for r in rows if int(r["correct"]) == 0]

    def card(r):
        uri = _img_to_data_uri(r["path"], max_side=max_side)
        conf = f'{r["confidence"]:.3f}'
        badge = "✅ Correct" if int(r["correct"]) == 1 else "❌ Wrong"
        return f"""
        <div class="card">
          <div class="imgwrap">{('<img src="'+uri+'"/>') if uri else '<div class="missing">Image missing</div>'}</div>
          <div class="meta">
            <div class="badge">{badge}</div>
            <div><b>True:</b> {r["true"]}</div>
            <div><b>Pred:</b> {r["pred"]}</div>
            <div><b>Conf:</b> {conf}</div>
            <div class="path">{r["path"]}</div>
          </div>
        </div>
        """

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 18px; }}
  h1 {{ margin: 0 0 8px 0; }}
  h2 {{ margin-top: 26px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }}
  .card {{ border: 1px solid #ddd; border-radius: 10px; overflow: hidden; background: #fff; }}
  .imgwrap {{ width: 100%; height: 220px; display:flex; align-items:center; justify-content:center; background:#fafafa; }}
  img {{ max-width: 100%; max-height: 100%; }}
  .meta {{ padding: 10px 12px; font-size: 14px; line-height: 1.35; }}
  .badge {{ font-weight: bold; margin-bottom: 6px; }}
  .path {{ color:#666; font-size: 11px; margin-top: 6px; word-break: break-all; }}
  .missing {{ color:#999; font-size: 12px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p>This page shows <b>Correct</b> (pred==true) and <b>Wrong</b> predictions (pred!=true). The model confidence is max softmax probability.</p>

<h2>✅ Correct predictions ({len(correct)})</h2>
<div class="grid">
{''.join(card(r) for r in correct)}
</div>

<h2>❌ Wrong predictions ({len(wrong)})</h2>
<div class="grid">
{''.join(card(r) for r in wrong)}
</div>

</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


def write_csv(rows: List[Dict[str, Any]], out_csv: str):
    fields = ["path", "true_idx", "true", "pred_idx", "pred", "confidence", "correct"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _find_best_test_metrics_json(ckpt_path: str) -> Optional[str]:
    """Try to locate the 'best test metrics' JSON produced by sweep/test scripts.

    Common filenames in your runs:
      - best_test_metrics.json
      - test_metrics.json
      - eval_test.json
    """
    d = os.path.dirname(os.path.abspath(ckpt_path))
    candidates = [
        os.path.join(d, "best_test_metrics.json"),
        os.path.join(d, "test_metrics.json"),
        os.path.join(d, "eval_test.json"),
        os.path.join(d, "viz_test", "test_metrics.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _format_metrics_for_title(m: Dict[str, Any]) -> str:
    """Pick a few common keys if present and format short."""
    # Accept both flat and nested dicts.
    if "test" in m and isinstance(m["test"], dict):
        m = m["test"]
    keys = [
        "f1_macro",
        "acc",
        "auroc",
        "auroc_ovr_macro",
        "bacc",
        "loss",
        "log_loss",
    ]
    parts = []
    for k in keys:
        if k in m:
            try:
                parts.append(f"{k}={float(m[k]):.4f}")
            except Exception:
                parts.append(f"{k}={m[k]}")
    return " | ".join(parts)


def export_images(rows: List[Dict[str, Any]], outdir: str, max_side: int = 512) -> Tuple[str, str]:
    """Export thumbnails into two folders: correct/ and wrong/."""
    correct_dir = os.path.join(outdir, "correct")
    wrong_dir = os.path.join(outdir, "wrong")
    _ensure_dir(correct_dir)
    _ensure_dir(wrong_dir)

    def safe_name(s: str) -> str:
        s = s.replace("/", "_").replace("\\", "_")
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)

    for idx, r in enumerate(rows):
        src = str(r.get("path", ""))
        if not src or not os.path.isfile(src):
            continue
        dst_dir = correct_dir if int(r.get("correct", 0)) == 1 else wrong_dir
        true_lbl = safe_name(str(r.get("true", "")))
        pred_lbl = safe_name(str(r.get("pred", "")))
        conf = r.get("confidence", 0.0)
        try:
            conf_s = f"{float(conf):.3f}"
        except Exception:
            conf_s = "na"
        base = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(dst_dir, f"{idx:05d}_{base}_T-{true_lbl}_P-{pred_lbl}_C-{conf_s}.jpg")
        try:
            img = Image.open(src).convert("RGB")
            w, h = img.size
            scale = min(1.0, float(max_side) / max(w, h))
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)))
            img.save(dst, format="JPEG", quality=90)
        except Exception:
            continue
    return correct_dir, wrong_dir


def _load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    try:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _infer_best_test_metrics_path(ckpt_path: str) -> str:
    """Try to find best_test_metrics.json near the checkpoint."""
    base = os.path.dirname(os.path.expandvars(ckpt_path))
    cand = os.path.join(base, "best_test_metrics.json")
    if os.path.exists(cand):
        return cand
    # sometimes metrics are saved inside viz dir or run dir; walk 1-level up
    cand2 = os.path.join(os.path.dirname(base), "best_test_metrics.json")
    if os.path.exists(cand2):
        return cand2
    return cand  # default location (may not exist)


def _export_sorted_lists(rows: List[Dict[str, Any]], outdir: str) -> Tuple[str, str]:
    """Write human-readable lists of correct/incorrect predictions."""
    correct_path = os.path.join(outdir, "correct_predictions.txt")
    wrong_path = os.path.join(outdir, "wrong_predictions.txt")

    correct = [r for r in rows if int(r.get("correct", 0)) == 1]
    wrong = [r for r in rows if int(r.get("correct", 0)) == 0]

    # Sort by confidence descending to quickly see strong vs weak cases
    correct.sort(key=lambda r: float(r.get("confidence", 0.0)), reverse=True)
    wrong.sort(key=lambda r: float(r.get("confidence", 0.0)), reverse=True)

    def fmt(r: Dict[str, Any]) -> str:
        return (
            f"conf={float(r.get('confidence', 0.0)):.4f} | "
            f"true={r.get('true','')} | pred={r.get('pred','')} | "
            f"path={r.get('path','')}"
        )

    with open(correct_path, "w", encoding="utf-8") as f:
        f.write("# Correct predictions (sorted by confidence desc)\n")
        for r in correct:
            f.write(fmt(r) + "\n")

    with open(wrong_path, "w", encoding="utf-8") as f:
        f.write("# Wrong predictions (sorted by confidence desc)\n")
        for r in wrong:
            f.write(fmt(r) + "\n")

    return correct_path, wrong_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="config yaml")
    ap.add_argument("--ckpt", required=True, help="path to best_trainable.pt (or any trainable checkpoint)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--outdir", default="", help="output directory (default: alongside ckpt)")
    ap.add_argument("--max_items", type=int, default=200, help="max samples to visualize (0 = all)")
    ap.add_argument("--max_side", type=int, default=256, help="thumbnail max side in pixels (HTML gallery)")
    ap.add_argument(
        "--export_thumbs",
        action="store_true",
        help="also export JPG thumbnails into outdir/correct and outdir/wrong",
    )
    ap.add_argument(
        "--thumb_side",
        type=int,
        default=512,
        help="thumbnail max side in pixels for exported JPGs (when --export_thumbs)",
    )
    ap.add_argument(
        "--metrics_json",
        default="",
        help="optional path to best_test_metrics.json to show in title (default: auto-detect near ckpt)",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # dataloaders
    train_dl, val_dl, test_dl, classes, _train_ds = build_ham_dataloaders(cfg)
    loader = val_dl if args.split == "val" else (test_dl if test_dl is not None else val_dl)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # model
    model = build_model(cfg, num_classes=len(classes))
    ckpt = os.path.expandvars(args.ckpt)
    load_trainable_checkpoint(model, ckpt)
    model.to(device)

    # outdir
    outdir = args.outdir.strip()
    if not outdir:
        outdir = os.path.join(os.path.dirname(ckpt), f"viz_{args.split}")
    _ensure_dir(outdir)

    rows, y_true, y_pred = run_inference(model, loader, device, classes, max_items=args.max_items)

    out_csv = os.path.join(outdir, "predictions.csv")
    out_png = os.path.join(outdir, "confusion_matrix.png")
    out_html = os.path.join(outdir, "gallery.html")

    # Try to show "best test metrics" (commonly produced by your sweep/test scripts).
    metrics = None
    if args.split == "test":
        metrics_path = args.metrics_json.strip() or (_find_best_test_metrics_json(ckpt) or "")
        if metrics_path:
            metrics = _read_json(metrics_path)

    title = f"HAM10000 {args.split} predictions"
    if metrics:
        title = f"{title}  |  {_format_metrics_for_title(metrics)}"

    write_csv(rows, out_csv)
    save_confusion_matrix(y_true, y_pred, classes, out_png)
    write_gallery(rows, out_html, title=title, max_side=args.max_side)

    # Write plain-text lists of corrected/uncorrected predicted labels
    correct_txt, wrong_txt = _export_sorted_lists(rows, outdir)

    # Optionally export thumbnails into correct/ and wrong/
    correct_dir = wrong_dir = ""
    if args.export_thumbs:
        correct_dir, wrong_dir = export_images(rows, outdir, max_side=args.thumb_side)

    # quick summary
    acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    print(f"[VIZ] wrote: {out_csv}")
    print(f"[VIZ] wrote: {out_png}")
    print(f"[VIZ] wrote: {out_html}")
    print(f"[VIZ] wrote: {correct_txt}")
    print(f"[VIZ] wrote: {wrong_txt}")
    if args.export_thumbs:
        print(f"[VIZ] exported thumbs: {correct_dir}")
        print(f"[VIZ] exported thumbs: {wrong_dir}")
    if metrics:
        print(f"[VIZ] best-test-metrics: {_format_metrics_for_title(metrics)}")
    print(f"[VIZ] samples={len(rows)} acc={acc:.4f}")


if __name__ == "__main__":
    main()
