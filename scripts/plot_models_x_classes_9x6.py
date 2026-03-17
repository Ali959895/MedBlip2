#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import math
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import glob, os
import numpy as np
#Run
# python plot_models_x_classes_9x6.py   --npz_dir /scratch/ali95/kvasir_rocplots/True_Pred_labels/   --manifest /scratch/ali95/kvasir-dataset/kvasir_split_70_20_10/test.csv   --root /scratch/ali95/kvasir-dataset/kvasir_split_70_20_10/   --outdir /scratch/ali95/kvasir_rocplots/vis_models_9x6   --class_names "dyed-lifted-polyps,dye-less-polyps,normal-cecum,normal-pylorus,normal-z-line,polyps,ulcerative-colitis,esophagitis"   --models "BLIP2_OPT_2.7B,BLIP2_OPT_6.7B,BLIP2_T5_flant5xl,BLIP2_vitL,BLIP_base,CLIP_VIT-B-32,CLIP_VIT-L-14,RN50,RN50x16"   --class_ids "0,1,2,3,4,5,6,7"   --seed 10   --thumb 256 --dpi 300   --title "Predicted classes for all models"

# ---------------- DEFAULT RUN CONFIG ----------------
DEFAULTS = dict(
    npz_dir="/scratch/ali95/kvasir_rocplots/True_Pred_labels",
    manifest="/scratch/ali95/kvasir-dataset/kvasir_split_70_20_10/test.csv",
    root="/scratch/ali95/kvasir-dataset/kvasir_split_70_20_10",
    outdir="/scratch/ali95/kvasir_rocplots/vis_models_9x6",
    #class_names="dyed-lifted-polyps,dye-less-polyps,normal-cecum,normal-pylorus,normal-z-line,polyps,ulcerative-colitis,esophagitis",
    class_names="dyed-lifted-polyps,dye-less-polyps,normal-cecum,normal-pylorus",
    #models="MedBLIP2-OPT-2.7B,MedBLIP2-OPT-6.7B,BLIP2-T5-flant5xl,BLIP2-vitL,BLIP-base,CLIP-ViT-B-32,CLIP-ViT-L-14,RN50,RN50x16",
    models="MedBLIP2-OPT-2.7B,MedBLIP2-OPT-6.7B,BLIP-base,CLIP-ViT-B-32,RN50",
    #class_ids="0,1,2,3,4,5,6,7",
    class_ids="0,1,2,3",
    seed=10,
    thumb=256,
    dpi=50,
    fig_w=50.0,
    fig_h=90.0,
    title="Predicted classes of all models for Kvasir dataset",
)
# ---------------------------------------------------



def pick_best_blip_worst_clip(
    y_true_ref,
    model_to_preds,
    class_ids,
    good_models=("MedBLIP2-OPT-2.7B", "MedBLIP2-OPT-6.7B"),
    #bad_models=("CLIP-ViT-B-32", "CLIP-ViT-L-14","BLIP2-T5-flant5xl"),
    bad_models=("CLIP-ViT-B-32","RN50"),
    seed=0,
    min_other_wrong=1,
    min_other_correct=1,
):
    """
    Returns dict {class_id -> index}.

    Primary goal per class:
      - BOTH good_models correct
      - BOTH bad_models wrong
      - Others mixed: at least min_other_wrong wrong and min_other_correct correct (among remaining models)

    Then progressively relax constraints if no sample satisfies them.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true_ref, dtype=int)

    available = set(model_to_preds.keys())
    for m in (*good_models, *bad_models):
        if m not in available:
            raise KeyError(f"Model '{m}' not in model_to_preds keys. Available: {sorted(available)}")

    other_models = [m for m in model_to_preds.keys() if (m not in good_models and m not in bad_models)]
    chosen = {}

    for c in class_ids:
        pool = np.where(y == int(c))[0]
        if len(pool) == 0:
            raise ValueError(f"No samples found for class {c}")

        # correctness arrays over pool
        good_ok = []
        for gm in good_models:
            good_ok.append((np.asarray(model_to_preds[gm], dtype=int)[pool] == y[pool]))
        good_ok = np.vstack(good_ok)  # (G, pool)

        bad_ok = []
        for bm in bad_models:
            bad_ok.append((np.asarray(model_to_preds[bm], dtype=int)[pool] == y[pool]))
        bad_ok = np.vstack(bad_ok)    # (B, pool)

        others_ok = []
        for om in other_models:
            others_ok.append((np.asarray(model_to_preds[om], dtype=int)[pool] == y[pool]))
        others_ok = np.vstack(others_ok) if len(other_models) else None

        good_all = good_ok.all(axis=0)                 # both BLIP correct
        bad_all_wrong = (~bad_ok).all(axis=0)          # both CLIP wrong

        if others_ok is not None:
            n_other_correct = others_ok.sum(axis=0)
            n_other_wrong = (others_ok.shape[0] - n_other_correct)
        else:
            n_other_correct = np.zeros(len(pool), dtype=int)
            n_other_wrong = np.zeros(len(pool), dtype=int)

        def choose(mask):
            idxs = pool[np.where(mask)[0]]
            if len(idxs) == 0:
                return None
            return int(rng.choice(idxs, size=1, replace=False)[0])

        # Tier 1 (ideal): BLIP correct AND both CLIP wrong AND others mixed
        mask1 = good_all & bad_all_wrong & (n_other_wrong >= min_other_wrong) & (n_other_correct >= min_other_correct)
        pick = choose(mask1)
        if pick is not None:
            chosen[int(c)] = pick
            continue

        # Tier 2: BLIP correct AND both CLIP wrong (ignore others mix)
        mask2 = good_all & bad_all_wrong
        pick = choose(mask2)
        if pick is not None:
            chosen[int(c)] = pick
            continue

        # Tier 3: BLIP correct AND at least ONE of the CLIP models wrong
        bad_any_wrong = (~bad_ok).any(axis=0)
        mask3 = good_all & bad_any_wrong
        pick = choose(mask3)
        if pick is not None:
            chosen[int(c)] = pick
            continue

        # Tier 4: at least ONE BLIP correct AND both CLIP wrong
        good_any = good_ok.any(axis=0)
        mask4 = good_any & bad_all_wrong
        pick = choose(mask4)
        if pick is not None:
            chosen[int(c)] = pick
            continue

        # Tier 5 fallback: random sample from this class
        chosen[int(c)] = int(rng.choice(pool, size=1, replace=False)[0])

    return chosen



def pick_indices_per_model_class(y_true_ref, model_names, class_ids, seed=0, pick="random"):
    """
    Returns dict: model_name -> {class_id -> index_in_dataset}
    If pick="first": deterministic first occurrence (same across runs)
    If pick="random": random per model and per class (changes with seed)
    """
    y = np.asarray(y_true_ref, dtype=int)
    class_to_pool = {}
    for c in class_ids:
        pool = np.where(y == c)[0]
        if len(pool) == 0:
            raise ValueError(f"No samples found for class {c}")
        class_to_pool[c] = pool

    out = {}
    for i, m in enumerate(model_names):
        # model-specific RNG stream (so different rows get different samples)
        rng = np.random.default_rng(seed + 1000 * i)
        out[m] = {}
        for c in class_ids:
            pool = class_to_pool[c]
            if pick == "first":
                out[m][c] = int(pool[0])
            else:
                out[m][c] = int(rng.choice(pool, size=1, replace=False)[0])
    return out


def sample_indices_per_class(y_true_all, class_ids, k, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true_all, dtype=int)

    picks = {}  # class_id -> list of row indices
    for c in class_ids:
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            raise ValueError(f"No samples found in manifest for class {c}")
        kk = min(k, len(idx))
        picks[c] = rng.choice(idx, size=kk, replace=False).tolist()
    return picks

def resolve_npz(npz_dir, model):
    # exact candidates
    cand = [
        os.path.join(npz_dir, f"{model}.npz"),
        os.path.join(npz_dir, f"test_preds_{model}.npz"),
    ]
    for c in cand:
        if os.path.exists(c):
            return c

    # fuzzy: any npz containing the model name (case-insensitive)
    files = glob.glob(os.path.join(npz_dir, "*.npz"))
    model_low = model.lower()
    hits = [f for f in files if model_low in os.path.basename(f).lower()]
    if not hits:
        return None
    # choose the shortest match (usually the cleanest)
    hits = sorted(hits, key=lambda p: len(os.path.basename(p)))
    return hits[0]

# -----------------------------
# Helpers
# -----------------------------
def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def _safe_open_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")

def _read_manifest_csv(manifest_csv: str):
    df = pd.read_csv(manifest_csv)

    cand = ["path", "filepath", "file", "image", "img", "rel_path", "relative_path"]
    path_col = None
    for c in cand:
        if c in df.columns:
            path_col = c
            break
    if path_col is None:
        raise KeyError(
            f"Could not find an image path column in {manifest_csv}. "
            f"Tried {cand}. Columns={list(df.columns)}"
        )

    paths = df[path_col].astype(str).tolist()
    return paths, df

def _resolve_paths(paths, root: str):
    out = []
    for p in paths:
        out.append(p if os.path.isabs(p) else os.path.join(root, p))
    return out

def _load_npz(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)

    def pick(*names):
        for n in names:
            if n in keys:
                return z[n]
        return None

    y_true = pick("y_true", "true", "labels_true", "targets", "gt")
    y_pred = pick("y_pred", "pred", "labels_pred", "preds")

    if y_true is None or y_pred is None:
        raise KeyError(f"{npz_path} must contain y_true and y_pred (or aliases). Found keys={z.files}")

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(f"{npz_path}: len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}")

    return y_true, y_pred

def _clean_model_name(filename: str):
    # test_preds_<MODEL>.npz  -> <MODEL>
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    stem = re.sub(r"^test_preds_", "", stem)
    return stem

def _name(idx, class_names):
    idx = int(idx)
    if class_names and 0 <= idx < len(class_names):
        return str(class_names[idx])
    return str(idx)

#def _pick_one_index_per_class(y_true_ref, class_ids, seed=0, pick="first"):
#    y = np.asarray(y_true_ref, dtype=int)
#    rng = np.random.default_rng(seed)
#    chosen = {}
#
#    for c in class_ids:
#        pool = np.where(y == c)[0]
#        if len(pool) == 0:
#            raise ValueError(f"No samples found for class {c}")
#        if pick == "first":
#            chosen[c] = int(pool[0])
#        elif pick == "random":
#            chosen[c] = int(rng.choice(pool, size=1, replace=False)[0])
#        else:
#            raise ValueError(f"pick must be 'first' or 'random', got {pick}")
#    return chosen
def _pick_one_index_per_class(y_true_ref, class_ids, seed=0, pick="first"):
    import numpy as np
    y = np.asarray(y_true_ref, dtype=int)
    rng = np.random.default_rng(seed)
    chosen = {}
    for c in class_ids:
        pool = np.where(y == c)[0]
        if len(pool) == 0:
            raise ValueError(f"No samples found for class {c}")
        if pick == "first":
            chosen[c] = int(pool[0])
        elif pick == "random":
            chosen[c] = int(rng.choice(pool, size=1, replace=False)[0])
        else:
            raise ValueError(f"pick must be 'first' or 'random', got {pick}")
    return chosen


# -----------------------------
# Main plotting: 9 models x 6 classes
# -----------------------------
def plot_models_by_class_9x6(
    img_paths,
    model_to_preds,
    y_true_ref,
    class_ids,
    class_names,
    out_png,
    out_eps,
    title="Model comparison (9x6)",
    thumb=256,
    cell_w=8.6, cell_h=8.6, fig_w=30.0, fig_h=15.0,
    dpi=50,
    seed=0,
    pick="first",   # "first" or "random"
):

    model_names = list(model_to_preds.keys())
    n_rows = len(model_names)
    n_cols = len(class_ids)

    # choose ONE sample index per class (same for all models)
    chosen = pick_best_blip_worst_clip(
        y_true_ref=y_true_ref,
        model_to_preds=model_to_preds,
        class_ids=class_ids,
        good_models=("MedBLIP2-OPT-2.7B", "MedBLIP2-OPT-6.7B"),
        #bad_models=("CLIP-ViT-B-32", "CLIP-ViT-L-14", "BLIP2-T5-flant5xl"),
        bad_models=("CLIP-ViT-B-32","RN50"),
        seed=seed,
        min_other_wrong=1,
        min_other_correct=1,
    )

    # figure sizing tuned for papers
    # figure sizing
    # if fig_w/fig_h provided (>0), they override the auto computation
    auto_w = n_cols * float(cell_w) + 0.55
    auto_h = n_rows * float(cell_h) + 1.0
    fig_w = float(fig_w) if float(fig_w) > 0 else auto_w
    fig_h = float(fig_h) if float(fig_h) > 0 else auto_h


    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    # tighter left margin to give more room to images
    plt.subplots_adjust(
        left=0.1, right=0.98, top=0.94, bottom=0.09,
        wspace=0.10, hspace=0.002
    )
    fig.suptitle(title, fontsize=44, fontweight="bold", y=0.93)

    # Column headers: class names
#    for j, c in enumerate(class_ids):
#        col_title = _name(c, class_names)
#        axes[0, j].set_title(col_title, fontsize=12, fontweight="bold", pad=10)

    # Row labels: rotated model names on the left, centered with each row
    # Place them in figure coordinates so they don't consume subplot space.
    x_text = 0.07  # move left/right here (smaller => closer to edge)
    for i, m in enumerate(model_names):
        # compute row vertical center using the first axis in the row
        ax0 = axes[i, 0]
        bb = ax0.get_position()
        y_mid = 0.5 * (bb.y0 + bb.y1)

        fig.text(
            x_text, y_mid, m,
            rotation=90,
            va="center", ha="center",
            fontsize=42, fontweight="bold"
        )

    # Draw cells
    for i, m in enumerate(model_names):
        y_pred = model_to_preds[m]

        for j, c in enumerate(class_ids):
            ax = axes[i, j]
            ax.axis("off")

            idx = chosen[int(c)] if isinstance(chosen, dict) else chosen[c]
            if idx is None:
                ax.text(0.5, 0.5, f"No samples\nfor class {c}", ha="center", va="center")
                continue

            p = img_paths[idx]
            im = _safe_open_rgb(p)
            if thumb and int(thumb) > 0:
                im = im.resize((int(thumb), int(thumb)), Image.BICUBIC)
            ax.imshow(im)

            yt = int(y_true_ref[idx])
            yp = int(y_pred[idx])
            ok = (yt == yp)

            # Border: green if correct else red
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(3.0)
                spine.set_edgecolor("#2ca02c" if ok else "#d62728")

            true_name = _name(yt, class_names)
            pred_name = _name(yp, class_names)

            label_box = dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=("#2ca02c" if ok else "#d62728"),
                linewidth=2.0
            )
            ax.text(
                0.5, -0.08,
                "Correct" if ok else "Wrong",
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=40, fontweight="bold",
                bbox=label_box
            )

            ax.text(
                0.5, -0.20,
                r"$\bf{True:}$ " + true_name + "\n" + r"$\bf{Pred:}$ " + pred_name,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=36, fontweight="bold"
            )

    _ensure_dir(os.path.dirname(out_png))
    #fig.savefig(out_png, bbox_inches="tight", pad_inches=0.15)
#    out_eps = os.path.splitext(out_eps)[0] + ".eps"
#    fig.savefig(out_eps, format="eps", bbox_inches="tight", pad_inches=0.15)

    out_png = os.path.splitext(out_png)[0] + ".png"
    fig.savefig(out_png, format="png", bbox_inches="tight", pad_inches=0.15)

    plt.close(fig)
    return out_png

#def plot_models_by_class_9x6(
#    img_paths,
#    model_to_preds,
#    y_true_ref,
#    class_ids,
#    class_names,
#    out_png,
#    title="Model comparison (9x6)",
#    thumb=256,
#    dpi=300,
#    seed=0,
#    pick="first",   # "first" or "random"
#):
#
#    model_names = list(model_to_preds.keys())
#    n_rows = len(model_names)
#    n_cols = len(class_ids)
#
#    # choose ONE sample index per class (same for all models)
#    chosen = pick_best_blip_worst_clip(
#        y_true_ref=y_true_ref,
#        model_to_preds=model_to_preds,
#        class_ids=class_ids,
#        good_models=("BLIP2_OPT_2.7B", "BLIP2_OPT_6.7B"),
#        bad_models=("CLIP_ViT-B-32", "CLIP_ViT-L-14","BLIP2_T5_flant5xl"),
#        seed=seed,
#        min_other_wrong=1,
#        min_other_correct=1,
#    )
#
#
#    #chosen = _pick_one_index_per_class(y_true_ref, class_ids, seed=0, pick="first")
#   #chosen = _pick_one_index_per_class(y_true_ref, class_ids, seed=seed, pick=pick)
##    chosen = _pick_one_index_per_class(
##        y_true_ref,
##        class_ids,
##        seed=seed,
##        pick="random",
##    )
#
#
#
#    # figure sizing tuned for papers
#    # each cell ~2.2 inch, plus small margins for under-text
#    cell_w = 2.2
#    cell_h = 2.6
#    fig_w = n_cols * cell_w + 1.2
#    fig_h = n_rows * cell_h + 1.0
#
#    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)
#    if n_rows == 1:
#        axes = np.expand_dims(axes, axis=0)
#    if n_cols == 1:
#        axes = np.expand_dims(axes, axis=1)
#
#    plt.subplots_adjust(left=0.14, right=0.995, top=0.93, bottom=0.05, wspace=0.10, hspace=0.35)
#    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
#
#    # Column headers: class names
#    for j, c in enumerate(class_ids):
#        col_title = _name(c, class_names)
#        axes[0, j].set_title(col_title, fontsize=12, fontweight="bold", pad=10)
#
#    # Row labels: model names on the left, centered with each row
#    for i, m in enumerate(model_names):
#        ax0 = axes[i, 0]
#        y_mid = (ax0.get_position().y0 + ax0.get_position().y1) / 2.0
#        x_left = ax0.get_position().x0 - 0.06
#        fig.text(
#            x_left, y_mid, m,
#            va="center", ha="right",
#            fontsize=12, fontweight="bold"
#        )
#
#    # Draw cells
#    for i, m in enumerate(model_names):
#        y_pred = model_to_preds[m]
#
#        for j, c in enumerate(class_ids):
#            ax = axes[i, j]
#            ax.axis("off")
#
#            #idx = chosen[int(c)]
#            #idx = chosen_map[m][c]
#            idx = chosen[c]
#
#            if idx is None:
#                ax.text(0.5, 0.5, f"No samples\nfor class {c}", ha="center", va="center")
#                continue
#
#            p = img_paths[idx]
#            im = _safe_open_rgb(p)
#            if thumb and int(thumb) > 0:
#                im = im.resize((int(thumb), int(thumb)), Image.BICUBIC)
#            ax.imshow(im)
#
#            yt = int(y_true_ref[idx])
#            yp = int(y_pred[idx])
#            ok = (yt == yp)
#
#            # Border: green if correct else red
#            for spine in ax.spines.values():
#                spine.set_visible(True)
#                spine.set_linewidth(3.0)
#                spine.set_edgecolor("#2ca02c" if ok else "#d62728")
#
#            # Under-text block (publication-friendly)
#            true_name = _name(yt, class_names)
#            pred_name = _name(yp, class_names)
#
#            # Small label box "Correct"/"Wrong"
#            label_box = dict(
#                boxstyle="round,pad=0.25",
#                facecolor="white",
#                edgecolor=("#2ca02c" if ok else "#d62728"),
#                linewidth=2.0
#            )
#            ax.text(
#                0.5, -0.08,
#                "Correct" if ok else "Wrong",
#                transform=ax.transAxes,
#                ha="center", va="top",
#                fontsize=9, fontweight="bold",
#                bbox=label_box
#            )
#
#            ax.text(
#                0.5, -0.20,
#                r"$\bf{True:}$ " + true_name + "\n" + r"$\bf{Pred:}$ " + pred_name,
#                transform=ax.transAxes,
#                ha="center", va="top",
#                fontsize=8
#            )
#
#    _ensure_dir(os.path.dirname(out_png))
#    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
#    plt.close(fig)
#    return out_png


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=str, default=DEFAULTS["npz_dir"])
    ap.add_argument("--manifest", type=str, default=DEFAULTS["manifest"])
    ap.add_argument("--root", type=str, default=DEFAULTS["root"])
    ap.add_argument("--outdir", type=str, default=DEFAULTS["outdir"])
    
    ap.add_argument("--class_names", type=str, default=DEFAULTS["class_names"])
    ap.add_argument("--models", type=str, default=DEFAULTS["models"])
    ap.add_argument("--class_ids", type=str, default=DEFAULTS["class_ids"])
    
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--thumb", type=int, default=DEFAULTS["thumb"])
    ap.add_argument("--dpi", type=int, default=DEFAULTS["dpi"])
    
    ap.add_argument("--fig_w", type=float, default=DEFAULTS["fig_w"])
    ap.add_argument("--fig_h", type=float, default=DEFAULTS["fig_h"])
    
    ap.add_argument("--title", type=str, default=DEFAULTS["title"])

    ap.add_argument("--samples_per_class", type=int, default=1,
                    help="How many images to sample per class (default 1).")
    ap.add_argument("--cell_w", type=float, default=1.2, help="Cell width in inches.")
    ap.add_argument("--cell_h", type=float, default=0.6, help="Cell height in inches.")
    
    
    args = ap.parse_args()
    # ---- models list ----
    if getattr(args, "models", None) and args.models.strip():
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        # auto-discover all .npz files in npz_dir (stems)
        models = []
        for fn in sorted(os.listdir(args.npz_dir)):
            if fn.endswith(".npz") and not fn.startswith("."):
                models.append(os.path.splitext(fn)[0])
    
    if not models:
        raise RuntimeError(f"No models specified and no .npz files found in: {args.npz_dir}")

    _ensure_dir(args.outdir)

    class_names = [s.strip() for s in args.class_names.split(",")] if args.class_names.strip() else None

    manifest_paths, _df = _read_manifest_csv(args.manifest)
    img_paths = _resolve_paths(manifest_paths, args.root)

    # collect npz files
    npz_files = []
    y_true_ref = None
    model_to_preds = {}
    for model in models:
        npz_path = resolve_npz(args.npz_dir, model)
        yt, yp = _load_npz(npz_path)
        if y_true_ref is None:
            y_true_ref = yt
        model_to_preds[model] = yp
        if npz_path is None:
            raise FileNotFoundError(f"Could not find NPZ for model '{model}' in {args.npz_dir}")
        npz_files.append(npz_path)


    # load all preds
    model_to_preds = {}
    y_true_ref = None
    for f in npz_files:
        yt, yp = _load_npz(f)
        if y_true_ref is None:
            y_true_ref = yt
        else:
            if len(yt) != len(y_true_ref) or not np.array_equal(yt, y_true_ref):
                # strict check: your saved y_true should match across models if they used same test order
                raise ValueError(
                    f"y_true mismatch across NPZ files.\n"
                    f"Reference: {npz_files[0]}\n"
                    f"Mismatch:  {f}\n"
                    f"This usually means NPZs were saved with different test order or different test split."
                )
        model_to_preds[_clean_model_name(f)] = yp

    # sanity length check vs manifest
    if len(img_paths) != len(y_true_ref):
        raise ValueError(
            f"Manifest rows ({len(img_paths)}) != y_true length ({len(y_true_ref)}). "
            "Use the exact same test.csv that created the NPZ files."
        )

#    picks = sample_indices_per_class(
#        y_true_all=y_true_manifest,   # your manifest true labels
#        class_ids=class_ids,
#        k=args.samples_per_class,
#        seed=args.seed)
      

    # pick 6 classes
    if args.class_ids.strip():
        class_ids = [int(x.strip()) for x in args.class_ids.split(",") if x.strip()]
    else:
        # default: first 6 classes
        class_ids = list(range(n_cols ))

    if len(class_ids) < 1:
        raise ValueError(f"--class_ids must contain exactly 6 class indices. Got: {class_ids}")

    # output
    out_png = os.path.join(args.outdir, "models_9x6_compare.png")
    out_eps = os.path.join(args.outdir, "models_9x6_compare.eps")
    title = args.title.strip() if args.title.strip() else "Model comparison (9 models x 6 classes)"

    plot_models_by_class_9x6(
    img_paths=img_paths,              # resolved image paths from manifest
    model_to_preds=model_to_preds,
    y_true_ref=y_true_ref,
    class_ids=class_ids,
    class_names=class_names,
    out_png=out_png,
    out_eps=out_eps,
    title=args.title,
    thumb=args.thumb,
    dpi=args.dpi,
    seed=args.seed,
    pick="random",
    cell_w=args.cell_w,
    cell_h=args.cell_h,
    fig_w=args.fig_w,
    fig_h=args.fig_h,
      )


    print(f"[OK] Saved: {out_png}")
    print(f"[OK] Saved: {out_eps}")
    print("[INFO] Rows=models, Cols=classes. Borders: green=correct, red=wrong.")


if __name__ == "__main__":
    main()


#Run
#python plot_models_x_classes_9x6.py --seed 1

