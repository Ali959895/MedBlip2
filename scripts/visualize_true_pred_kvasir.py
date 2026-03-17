#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import math
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# -----------------------------
# Utilities
# -----------------------------
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_pub_compare_2x6(
    n_correct, 
    n_wrong,
    paths_correct,
    y_true_correct,
    y_pred_correct,
    paths_wrong,
    y_true_wrong,
    y_pred_wrong,
    class_names,
    out_png,
    title="Correct vs Wrong (2x6)",
    thumb=256,
    dpi=300,
    left_label_x=0.055,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    def _to_int(x):
        # x can be python int, numpy scalar, or a 0-d array
        if isinstance(x, (np.ndarray,)):
            x = np.asarray(x).reshape(-1)
            if len(x) == 0:
                return 0
            return int(x[0])
        return int(x)

    def _name(idx):
        idx = _to_int(idx)
        if 0 <= idx < len(class_names):
            return str(class_names[idx])
        return str(idx)

    def _load_rgb(p):
        im = Image.open(p).convert("RGB")
        if thumb is not None and thumb > 0:
            im = im.resize((thumb, thumb), Image.BICUBIC)
        return im

    # ensure arrays are flat
    y_true_correct = np.asarray(y_true_correct).reshape(-1)
    y_pred_correct = np.asarray(y_pred_correct).reshape(-1)
    y_true_wrong   = np.asarray(y_true_wrong).reshape(-1)
    y_pred_wrong   = np.asarray(y_pred_wrong).reshape(-1)

    # enforce 6 per row (but allow fewer if not available)
    nC = min(6, len(paths_correct))
    nW = min(6, len(paths_wrong))

    paths_correct = list(paths_correct)[:nC]
    paths_wrong   = list(paths_wrong)[:nW]
    y_true_correct = y_true_correct[:nC]
    y_pred_correct = y_pred_correct[:nC]
    y_true_wrong   = y_true_wrong[:nW]
    y_pred_wrong   = y_pred_wrong[:nW]

    # Figure layout: give extra bottom margin for text under each image
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.5), dpi=dpi)
    plt.subplots_adjust(left=0.08, right=0.995, top=0.90, bottom=0.12, wspace=0.08, hspace=0.30)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.965)

    # Left row labels
    # ---- Big, clear row labels on the left (auto-centered using axes positions) ----
    # ---- Left-side row labels centered using actual axes geometry ----
    box = dict(facecolor="white", edgecolor="black", linewidth=1.0, pad=6)
    
    # Row 0 and Row 1 axes (first column is enough)
    ax_top = axes[0, 0]
    ax_bot = axes[1, 0]
    
    # Center y of each row in *figure* coordinates
    y_top = (ax_top.get_position().y0 + ax_top.get_position().y1) / 2.0
    y_bot = (ax_bot.get_position().y0 + ax_bot.get_position().y1) / 2.0
    
    # Place labels slightly left of the subplot grid (x just before the left edge)
    x_left = ax_top.get_position().x0 - 0.015  # adjust 0.02–0.05 as you like
    
    fig.text(
        x_left, y_top, f"Correct\n(n={n_correct})",
        rotation=90, va="center", ha="center",
        fontsize=16, fontweight="bold", bbox=box
    )
    fig.text(
        x_left, y_bot, f"Wrong\n(n={n_wrong})",
        rotation=90, va="center", ha="center",
        fontsize=16, fontweight="bold", bbox=box
    )



    # Helper to draw one cell
    def draw_cell(ax, img_path, yt, yp, ok: bool):
        ax.imshow(_load_rgb(img_path))
        ax.axis("off")
    
        # ---- marker (NO emoji; use colored label box) ----
        marker = "Correct" if ok else "Wrong"
        box = dict(
            boxstyle="round,pad=0.25",
            facecolor=("white"),
            edgecolor=("green" if ok else "red"),
            linewidth=2.0
        )
    
        ax.text(
            0.5, -0.07, marker,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=10, fontweight="bold",
            bbox=box
        )
    
        # ---- True / Pred lines (bold True/Pred) ----
        true_name = _name(yt)
        pred_name = _name(yp)
    
        ax.text(
            0.5, -0.17,
            r"$\bf{True:}$ " + true_name + "\n" + r"$\bf{Pred:}$ " + pred_name,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9
        )


    # Fill Correct row
    for j in range(6):
        ax = axes[0, j]
        if j < nC:
            draw_cell(ax, paths_correct[j], y_true_correct[j], y_pred_correct[j], ok=True)
        else:
            ax.axis("off")

    # Fill Wrong row
    for j in range(6):
        ax = axes[1, j]
        if j < nW:
            draw_cell(ax, paths_wrong[j], y_true_wrong[j], y_pred_wrong[j], ok=False)
        else:
            ax.axis("off")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


    
def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _safe_open_rgb(path: str) -> Image.Image:
    # PIL is fast enough for grids; convert to RGB for consistent rendering
    with Image.open(path) as im:
        return im.convert("RGB")


def _read_manifest_csv(manifest_csv: str):
    """
    Expects a CSV with at least:
      - an image path column (common: 'path', 'filepath', 'image', 'img', 'rel_path', etc.)
      - optional label column (not required for this script since labels come from npz)
    """
    df = pd.read_csv(manifest_csv)

    # try to auto-detect path column
    cand = ["path", "filepath", "file", "image", "img", "rel_path", "relative_path"]
    path_col = None
    for c in cand:
        if c in df.columns:
            path_col = c
            break
    if path_col is None:
        raise KeyError(f"Could not find an image path column in {manifest_csv}. "
                       f"Tried: {cand}. Columns={list(df.columns)}")

    paths = df[path_col].astype(str).tolist()
    return paths, df


def _resolve_paths(paths, root: str):
    out = []
    for p in paths:
        if os.path.isabs(p):
            out.append(p)
        else:
            out.append(os.path.join(root, p))
    return out


def _load_npz_preds(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)

    # accept common naming variants
    def pick(*names):
        for n in names:
            if n in keys:
                return z[n]
        return None

    y_true = pick("y_true", "true", "labels_true", "targets", "gt")
    y_pred = pick("y_pred", "pred", "labels_pred", "preds")
    # optional confidence/prob
    y_prob = pick("y_prob", "probs", "prob", "p")

    if y_true is None or y_pred is None:
        raise KeyError(f"{npz_path} must contain y_true and y_pred (or aliases). Found keys={z.files}")

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    ok_mask = (y_true == y_pred)
    n_correct = int(ok_mask.sum())
    n_wrong = int((~ok_mask).sum())
    if y_prob is not None:
        y_prob = np.asarray(y_prob)

    if len(y_true) != len(y_pred):
        raise ValueError(f"len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)} in {npz_path}")

    return y_true, y_pred, y_prob, n_correct, n_wrong


def _sample_indices(indices, max_n: int, seed: int = 0):
    indices = np.asarray(indices, dtype=int)
    if max_n <= 0 or len(indices) <= max_n:
        return indices
    rng = np.random.default_rng(seed)
    sel = rng.choice(indices, size=max_n, replace=False)
    return np.sort(sel)


def _stratified_wrong_indices(y_true, y_pred, max_wrong: int, seed: int = 0):
    """
    Pick wrong indices but diverse across confusion pairs (t,p).
    This produces a more publication-friendly set than pure random.
    """
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) <= max_wrong:
        return wrong

    rng = np.random.default_rng(seed)

    # group by (true, pred)
    pairs = {}
    for i in wrong:
        key = (int(y_true[i]), int(y_pred[i]))
        pairs.setdefault(key, []).append(int(i))

    # sort pairs by frequency (show the most common confusions first)
    pair_items = sorted(pairs.items(), key=lambda kv: len(kv[1]), reverse=True)

    picked = []
    # round-robin from pairs
    while len(picked) < max_wrong:
        progressed = False
        for (t, p), idxs in pair_items:
            if len(picked) >= max_wrong:
                break
            if not idxs:
                continue
            progressed = True
            # pop one random from this pair bucket
            j = rng.integers(0, len(idxs))
            picked.append(idxs.pop(j))
        if not progressed:
            break

    return np.array(sorted(picked), dtype=int)


# -----------------------------
# Publication plotting
# -----------------------------
def _set_pub_style(font_scale: float = 1.0):
    # simple, consistent "paper" style (no seaborn)
    base = 10.0 * float(font_scale)
    plt.rcParams.update({
        "font.size": base,
        "axes.titlesize": base + 3,
        "axes.labelsize": base + 1,
        "figure.titlesize": base + 5,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    })


def _draw_grid(
    img_paths,
    y_true,
    y_pred,
    class_names,
    idxs,
    title,
    out_path_png,
    thumb=224,
    cols=6,
    dpi=300,
    font_scale=1.0,
    border=False,
    border_lw=3.0,
    border_color=None,
):
    """
    border: if True, draws a border around each tile.
    border_color: if None, use border_color per correctness (handled outside).
    """
    _set_pub_style(font_scale=font_scale)

    idxs = np.asarray(idxs, dtype=int)
    n = len(idxs)
    if n == 0:
        print(f"[WARN] No samples to plot for: {title}")
        return None

    cols = int(cols)
    rows = int(math.ceil(n / cols))

    # figure size: ~2.1 inches per tile works well for papers
    tile_in = 2.1
    fig_w = cols * tile_in
    fig_h = rows * tile_in + 0.6  # small room for title

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(rows, cols)

    for k, idx in enumerate(idxs):
        r, c = divmod(k, cols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        p = img_paths[idx]
        try:
            im = _safe_open_rgb(p)
            if thumb and int(thumb) > 0:
                im = im.resize((int(thumb), int(thumb)))
            ax.imshow(im)
        except Exception as e:
            ax.text(0.5, 0.5, f"LOAD FAIL\n{os.path.basename(p)}\n{e}",
                    ha="center", va="center")
            continue

        t = int(y_true[idx])
        pr = int(y_pred[idx])
        tname = class_names[t] if (class_names and t < len(class_names)) else f"c{t}"
        pname = class_names[pr] if (class_names and pr < len(class_names)) else f"c{pr}"

        # compact title for papers
        ax.set_title(f"T: {tname}\nP: {pname}", pad=2)

        if border:
            # draw a visible border around the axes image
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(border_lw)
                spine.set_edgecolor(border_color if border_color is not None else "black")

    # hide empty slots
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

    fig.suptitle(title)

    fig.savefig(out_path_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return out_path_png


def _draw_compare_two_panel(
    img_paths,
    y_true,
    y_pred,
    class_names,
    correct_idxs,
    wrong_idxs,
    stem,
    out_path_png,
    cols=6,
    thumb=224,
    dpi=300,
    font_scale=1.0,
):
    """
    Single figure with two panels:
      (A) Correct predictions
      (B) Wrong predictions
    """
    _set_pub_style(font_scale=font_scale)

    correct_idxs = np.asarray(correct_idxs, dtype=int)
    wrong_idxs = np.asarray(wrong_idxs, dtype=int)

    nA = len(correct_idxs)
    nB = len(wrong_idxs)

    rowsA = int(math.ceil(max(nA, 1) / cols))
    rowsB = int(math.ceil(max(nB, 1) / cols))

    tile_in = 2.0
    fig_w = cols * tile_in
    # two panels stacked, each panel height based on its rows
    fig_h = (rowsA + rowsB) * tile_in + 1.2

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[rowsA, rowsB])

    def fill_panel(gs_panel, idxs, panel_title, border_color):
        sub = gs_panel.subgridspec(rowsA if panel_title.startswith("A") else rowsB, cols)
        idxs = list(idxs)
        for k, idx in enumerate(idxs):
            r, c = divmod(k, cols)
            ax = fig.add_subplot(sub[r, c])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            p = img_paths[idx]
            try:
                im = _safe_open_rgb(p)
                if thumb and int(thumb) > 0:
                    im = im.resize((int(thumb), int(thumb)))
                ax.imshow(im)
            except Exception as e:
                ax.text(0.5, 0.5, f"LOAD FAIL\n{os.path.basename(p)}\n{e}",
                        ha="center", va="center")
                continue

            t = int(y_true[idx])
            pr = int(y_pred[idx])
            tname = class_names[t] if (class_names and t < len(class_names)) else f"c{t}"
            pname = class_names[pr] if (class_names and pr < len(class_names)) else f"c{pr}"
            ax.set_title(f"T: {tname}\nP: {pname}", pad=2)

            # colored border to quickly distinguish panels (publication-friendly)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.5)
                spine.set_edgecolor(border_color)

        # hide empty slots
        total = (rowsA if panel_title.startswith("A") else rowsB) * cols
        for k in range(len(idxs), total):
            r, c = divmod(k, cols)
            ax = fig.add_subplot(sub[r, c])
            ax.axis("off")

        # panel label
        fig.text(0.01, 0.98 if panel_title.startswith("A") else 0.49,
                 panel_title, fontsize=plt.rcParams["figure.titlesize"], fontweight="bold")

    # panel A
    fill_panel(outer[0], correct_idxs, f"A  Correct (n={nA})", border_color="#2ca02c")
    # panel B
    fill_panel(outer[1], wrong_idxs, f"B  Wrong (n={nB})", border_color="#d62728")

    fig.suptitle(f"{stem} - Correct vs Wrong predictions", y=1.01)
    fig.savefig(out_path_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return out_path_png


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="NPZ file containing y_true and y_pred (and optionally probs).")
    ap.add_argument("--manifest", required=True, help="CSV manifest for the test set (contains image paths).")
    ap.add_argument("--root", required=True, help="Root directory to resolve relative paths in manifest.")
    ap.add_argument("--outdir", required=True, help="Output directory for figures.")
    ap.add_argument("--class_names", default="", help="Comma-separated class names in label index order.")
    ap.add_argument("--max_wrong", type=int, default=48)
    ap.add_argument("--max_correct", type=int, default=48)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--thumb", type=int, default=224)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="", help="Figure title (optional)")

    # publication options
    ap.add_argument("--pub", action="store_true", help="Publication style (bigger fonts, tighter layout).")
    ap.add_argument("--font_scale", type=float, default=1.15, help="Font scaling for publication figures.")
    ap.add_argument("--compare", action="store_true",
                    help="Also save a single 2-panel figure comparing Correct vs Wrong.")
    ap.add_argument("--save_pdf", action="store_true",
                    help="Save a multipage PDF (correct + wrong + compare if enabled).")

    args = ap.parse_args()
    _ensure_dir(args.outdir)

    stem = os.path.splitext(os.path.basename(args.npz))[0].replace("test_preds_", "")
    class_names = [s.strip() for s in args.class_names.split(",")] if args.class_names.strip() else None

    y_true, y_pred,_, n_correct, n_wrong = _load_npz_preds(args.npz)

    manifest_paths, _df = _read_manifest_csv(args.manifest)
    img_paths = _resolve_paths(manifest_paths, args.root)

    if len(img_paths) != len(y_true):
        raise ValueError(
            f"Manifest rows ({len(img_paths)}) != y_true length ({len(y_true)}). "
            "You must use the exact same test.csv used to generate the NPZ."
        )

    correct_idx = np.where(y_true == y_pred)[0]
    wrong_idx = np.where(y_true != y_pred)[0]

    # sampling
    correct_sel = _sample_indices(correct_idx, args.max_correct, seed=args.seed)
    wrong_sel = _stratified_wrong_indices(y_true, y_pred, args.max_wrong, seed=args.seed)

    # file names
    correct_png = os.path.join(args.outdir, f"{stem}_correct_grid_pub.png")
    wrong_png = os.path.join(args.outdir, f"{stem}_wrong_grid_pub.png")
    compare_png = os.path.join(args.outdir, f"{stem}_compare_pub.png")
    pdf_path = os.path.join(args.outdir, f"{stem}_publication_figures.pdf")

    # style knobs
    font_scale = args.font_scale if args.pub else 1.0

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
    # Pick 6 correct + 6 wrong (or fewer if not enough)
    correct_idx = np.where(np.asarray(y_true).reshape(-1) == np.asarray(y_pred).reshape(-1))[0]
    wrong_idx   = np.where(np.asarray(y_true).reshape(-1) != np.asarray(y_pred).reshape(-1))[0]
    
    np.random.shuffle(correct_idx)
    np.random.shuffle(wrong_idx)
    
    correct_idx = correct_idx[:6]
    wrong_idx   = wrong_idx[:6]
    
    paths_correct = [img_paths[i] for i in correct_idx]
    paths_wrong   = [img_paths[i] for i in wrong_idx]
    
    y_true_correct = np.asarray(y_true).reshape(-1)[correct_idx]
    y_pred_correct = np.asarray(y_pred).reshape(-1)[correct_idx]
    y_true_wrong   = np.asarray(y_true).reshape(-1)[wrong_idx]
    y_pred_wrong   = np.asarray(y_pred).reshape(-1)[wrong_idx]
    
    plot_pub_compare_2x6(
        n_correct=n_correct , n_wrong=n_wrong,
        paths_correct=paths_correct,
        y_true_correct=y_true_correct,
        y_pred_correct=y_pred_correct,
        paths_wrong=paths_wrong,
        y_true_wrong=y_true_wrong,
        y_pred_wrong=y_pred_wrong,
        class_names=class_names,
        out_png=out_png,
        title=title,
        thumb=args.thumb,
        dpi=args.dpi,
        left_label_x=0.055,  # move "Correct/Wrong" a bit right; adjust if needed
    )

    
    # 1) Correct grid
    _draw_grid(
        img_paths=img_paths,
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        idxs=correct_sel,
        title=f"{stem} - CORRECT predictions (n={len(correct_idx)})",
        out_path_png=correct_png,
        thumb=args.thumb,
        cols=args.cols,
        dpi=args.dpi,
        font_scale=font_scale,
        border=True,
        border_lw=2.5,
        border_color="#2ca02c",
    )

    # 2) Wrong grid (stratified by confusion pairs)
    _draw_grid(
        img_paths=img_paths,
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        idxs=wrong_sel,
        title=f"{stem} - WRONG predictions (n={len(wrong_idx)})",
        out_path_png=wrong_png,
        thumb=args.thumb,
        cols=args.cols,
        dpi=args.dpi,
        font_scale=font_scale,
        border=True,
        border_lw=2.5,
        border_color="#d62728",
    )

    # 3) Compare 2-panel (optional)
    if args.compare:
        _draw_compare_two_panel(
            img_paths=img_paths,
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            correct_idxs=correct_sel,
            wrong_idxs=wrong_sel,
            stem=stem,
            out_path_png=compare_png,
            cols=args.cols,
            thumb=args.thumb,
            dpi=args.dpi,
            font_scale=font_scale,
        )

    # 4) Multipage PDF (optional)
    if args.save_pdf:
        with PdfPages(pdf_path) as pdf:
            for p in [correct_png, wrong_png, (compare_png if args.compare else None)]:
                if p and os.path.exists(p):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.axis("off")
                    ax.imshow(_safe_open_rgb(p))
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

        print(f"[OK] Saved multipage PDF: {pdf_path}")

    print(f"[OK] Saved:\n  {correct_png}\n  {wrong_png}")
    if args.compare:
        print(f"  {compare_png}")


if __name__ == "__main__":
    main()

 
 # Run it by this
    
#python visualize_true_pred_kvasir.py   --npz /scratch/ali95/kvasir_rocplots/True_Pred_labels/test_preds_BLIP2_OPT_2.7B.npz   --manifest /scratch/ali95/kvasir-dataset/kvasir_split_70_20_10/test.csv   --root /scratch/ali95/kvasir-dataset/kvasir_split_70_20_10/   --outdir /scratch/ali95/kvasir_rocplots/vis_BLIP2_OPT_2.7B   --class_names "dyed-lifted-polyps,dye-less-polyps,normal-cecum,normal-pylorus,normal-z-line,polyps,ulcerative-colitis,esophagitis"   --max_wrong 48 --max_correct 48 --cols 6 --thumb 224

