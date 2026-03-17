#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def _npz_get_any(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def read_manifest_csv(manifest_path):
    """
    Reads CSV and returns:
      - rel_paths: list[str] (relative or absolute paths)
      - labels: list[int] or None
    Tries common column names.
    """
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    # Detect path column
    path_cols = ["path", "rel_path", "filepath", "image", "img", "filename", "file"]
    label_cols = ["label", "y", "target", "class", "cls", "category"]

    header = rows[0].keys()
    path_col = next((c for c in path_cols if c in header), None)
    if path_col is None:
        raise RuntimeError(
            f"Could not find an image path column in manifest. "
            f"Tried {path_cols}. Found columns: {list(header)}"
        )

    label_col = next((c for c in label_cols if c in header), None)

    rel_paths = [r[path_col] for r in rows]
    labels = None
    if label_col is not None:
        labels = []
        for r in rows:
            v = r[label_col]
            try:
                labels.append(int(v))
            except Exception:
                # allow label names, but then we can't use numeric metrics here
                labels.append(v)

    return rel_paths, labels


def make_abs_paths(rel_paths, root):
    out = []
    for p in rel_paths:
        p = p.strip()
        if os.path.isabs(p):
            out.append(p)
        else:
            out.append(os.path.join(root, p))
    return out


def idx_to_name(i, class_names):
    if class_names is None:
        return str(int(i))
    i = int(i)
    if 0 <= i < len(class_names):
        return class_names[i]
    return str(i)


def pick_unique_true_classes(indices, y_true, max_n):
    """
    Pick up to max_n indices such that true classes are as diverse as possible.
    """
    picked = []
    seen = set()
    for idx in indices:
        t = int(y_true[idx])
        if t in seen:
            continue
        seen.add(t)
        picked.append(idx)
        if len(picked) >= max_n:
            break
    # If not enough, fill with remaining
    if len(picked) < max_n:
        for idx in indices:
            if idx in picked:
                continue
            picked.append(idx)
            if len(picked) >= max_n:
                break
    return picked


def load_image_safe(path, thumb):
    """
    Loads an image safely and returns an RGB PIL image resized for display.
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        # placeholder image if file missing/corrupt
        img = Image.new("RGB", (thumb, thumb), (255, 255, 255))
    if thumb is not None and thumb > 0:
        img = img.resize((thumb, thumb), Image.BILINEAR)
    return img


# -----------------------------
# Main plotting function
# -----------------------------
def plot_pub_compare_2x6(
    img_paths,
    y_true,
    y_pred,
    class_names,
    out_png,
    thumb=224,
    dpi=300,
    title=None,
):
    import numpy as np
    import os
    from PIL import Image
    import matplotlib.pyplot as plt

    def idx_to_name(i):
        i = int(i)
        if class_names is None:
            return str(i)
        if 0 <= i < len(class_names):
            return class_names[i]
        return str(i)

    def load_image_safe(path):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (thumb, thumb), (255, 255, 255))
        if thumb is not None and thumb > 0:
            img = img.resize((thumb, thumb), Image.BILINEAR)
        return img

    def pick_unique_true_classes(indices, y_true_arr, max_n):
        picked, seen = [], set()
        for idx in indices:
            t = int(y_true_arr[idx])
            if t in seen:
                continue
            seen.add(t)
            picked.append(idx)
            if len(picked) >= max_n:
                break
        if len(picked) < max_n:
            for idx in indices:
                if idx in picked:
                    continue
                picked.append(idx)
                if len(picked) >= max_n:
                    break
        return picked

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    correct_idx = np.where(y_true == y_pred)[0].tolist()
    wrong_idx = np.where(y_true != y_pred)[0].tolist()

    if len(correct_idx) == 0:
        raise RuntimeError("No correct predictions found.")
    if len(wrong_idx) == 0:
        raise RuntimeError("No wrong predictions found.")

    # diversify by TRUE class (publication-friendly)
    correct_pick = pick_unique_true_classes(correct_idx, y_true, max_n=6)
    wrong_pick = pick_unique_true_classes(wrong_idx, y_true, max_n=6)

    # Bigger figure and more headroom for text
    fig, axes = plt.subplots(2, 6, figsize=(20, 6.5))
    plt.subplots_adjust(left=0.08, right=0.995, top=0.90, bottom=0.05, wspace=0.15, hspace=0.35)

    if title is None:
        title = "Correct vs Wrong predictions"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # ---- Big, clear row labels on the left ----
    # Use a white box so it stays readable on any background
    # ---- Big, clear row labels on the left (centered to the image rows) ----
    box = dict(facecolor="white", edgecolor="black", linewidth=1.0, pad=6)
    
    # IMPORTANT: call plt.subplots_adjust(...) BEFORE this block
    # Use the first column axes to estimate each row's vertical center
    row0 = axes[0, 0].get_position()
    row1 = axes[1, 0].get_position()
    
    y_correct = 0.5 * (row0.y0 + row0.y1)
    y_wrong   = 0.5 * (row1.y0 + row1.y1)
    
    # Put label slightly left of the leftmost subplot
    left0 = axes[0, 0].get_position().x0
    x_lbl = max(0.0, left0 - 0.01)   # tweak -0.035 to move closer/farther
    
    fig.text(
        x_lbl, y_correct, "Correct",
        rotation=90, va="center", ha="center",
        fontsize=16, fontweight="bold", bbox=box
    )
    fig.text(
        x_lbl, y_wrong, "Wrong",
        rotation=90, va="center", ha="center",
        fontsize=16, fontweight="bold", bbox=box
    )


    # ---- Row 0: Correct ----
    for j, idx in enumerate(correct_pick):
        ax = axes[0, j]
        ax.axis("off")
        ax.imshow(load_image_safe(img_paths[idx]))

        tname = idx_to_name(y_true[idx])
        pname = idx_to_name(y_pred[idx])

        # For correct, still show True/Pred (they match), but it looks consistent
        ax.set_title(tname, fontsize=11, fontweight="bold", pad=6)
        ax.text(
            0.5, -0.10,
            f"True: {tname}\nPred: {pname}",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9
        )

    # ---- Row 1: Wrong ----
    for j, idx in enumerate(wrong_pick):
        ax = axes[1, j]
        ax.axis("off")
        ax.imshow(load_image_safe(img_paths[idx]))

        tname = idx_to_name(y_true[idx])
        pname = idx_to_name(y_pred[idx])

        ax.set_title(tname, fontsize=11, fontweight="bold", pad=6)
        ax.text(
            0.5, -0.10,
            f"True: {tname}\nPred: {pname}",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9
        )

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="NPZ containing y_true/y_pred (and optionally paths)")
    ap.add_argument("--manifest", required=True, help="CSV test manifest (must contain image path column)")
    ap.add_argument("--root", required=True, help="Root dir to prepend to manifest relative paths")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--class_names", default="", help="Comma-separated class names in correct index order")
    ap.add_argument("--thumb", type=int, default=224)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--title", default="", help="Figure title (optional)")
    args = ap.parse_args()

    # class names
    class_names = None
    if args.class_names.strip():
        class_names = [s.strip() for s in args.class_names.split(",")]

    # load manifest paths (we rely on manifest order)
    rel_paths, _ = read_manifest_csv(args.manifest)
    img_paths = make_abs_paths(rel_paths, args.root)

    # load predictions
    d = np.load(args.npz, allow_pickle=True)
    y_true = _npz_get_any(d, ["y_true", "true", "targets", "labels", "gt"])
    y_pred = _npz_get_any(d, ["y_pred", "pred", "preds", "pred_labels"])

    if y_true is None or y_pred is None:
        raise RuntimeError(
            f"Could not find y_true/y_pred in {args.npz}. "
            f"Available keys: {list(d.keys())}"
        )

    if len(y_true) != len(img_paths) or len(y_pred) != len(img_paths):
        raise RuntimeError(
            f"Length mismatch:\n"
            f"  manifest images: {len(img_paths)}\n"
            f"  y_true: {len(y_true)}\n"
            f"  y_pred: {len(y_pred)}\n"
            f"Make sure the NPZ was saved in the same order as test.csv."
        )

    stem = os.path.splitext(os.path.basename(args.npz))[0]
    out_png = os.path.join(args.outdir, f"{stem}_pub_correct_vs_wrong_2x6.png")

    title = args.title.strip() if args.title.strip() else f"Correct vs Wrong Predections for BLIP_OPT_2.7B"
    plot_pub_compare_2x6(
        img_paths=img_paths,
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        out_png=out_png,
        thumb=args.thumb,
        dpi=args.dpi,
        title=title,
    )

    print(f"[OK] Saved: {out_png}")


if __name__ == "__main__":
    main()
