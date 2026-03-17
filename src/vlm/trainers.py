# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.metrics import confusion_matrix
import math
from typing import Dict, Any, Optional, Sequence, Union
     
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
from sklearn.metrics import classification_report


import math
import matplotlib
matplotlib.use("Agg")  # safe on headless nodes
import matplotlib.pyplot as plt
def _safe_div(num, den):
    return float(num) / float(den) if den != 0 else float("nan")

import numpy as np

def binary_metrics_from_confusion(tp, tn, fp, fn, y_true_bin=None, y_score_pos=None):
    """
    Compute binary metrics from confusion counts.
    AUROC needs (y_true_bin, y_score_pos) and is optional.

    Args:
      tp, tn, fp, fn: ints
      y_true_bin: np.ndarray of shape [N] with 0/1
      y_score_pos: np.ndarray of shape [N] with scores for positive class (prob or logit)

    Returns:
      dict of metrics
    """
    # Basic rates
    acc  = _safe_div(tp + tn, tp + tn + fp + fn)
    prec = _safe_div(tp, tp + fp)          # PPV
    rec  = _safe_div(tp, tp + fn)          # recall
    spec = _safe_div(tn, tn + fp)          # TNR
    sens = _safe_div(tp, tp + fn)                             # sensitivity = recall

    # F1
    f1 = _safe_div(2 * prec * rec, prec + rec)

    # MCC (from confusion)
    # mcc = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = _safe_div((tp * tn - fp * fn), np.sqrt(denom) if denom > 0 else 0.0)

    # Cohen's Kappa (from confusion)
    total = tp + tn + fp + fn
    po = _safe_div(tp + tn, total)  # observed agreement
    pe = 0.0
    if total > 0:
        # expected agreement from marginals
        pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / float(total * total)
    kappa = _safe_div((po - pe), (1.0 - pe))

    # AUROC (one-vs-rest): requires scores, cannot be derived from confusion alone
    auroc_ovr = float("nan")

    if y_true_bin is None or y_score_pos is None:
        # You didn't pass scores/labels
        auroc_ovr = float("nan")
    else:
        from sklearn.metrics import roc_auc_score
        y_true_bin = np.asarray(y_true_bin).astype(np.int32).reshape(-1)
        y_score_pos = np.asarray(y_score_pos).astype(np.float64).reshape(-1)
    
        # basic sanity
        if y_true_bin.shape[0] != y_score_pos.shape[0]:
            print(f"[AUROC WARN] shape mismatch: y_true_bin={y_true_bin.shape} y_score_pos={y_score_pos.shape}", flush=True)
            auroc_ovr = float("nan")
        else:
            y_score_pos = np.nan_to_num(y_score_pos, nan=0.0, posinf=1.0, neginf=0.0)
    
            n_pos = int(y_true_bin.sum())
            n_neg = int((1 - y_true_bin).sum())
            s_min, s_max = float(y_score_pos.min()), float(y_score_pos.max())
    
            print(f"[AUROC DBG] n={len(y_true_bin)} pos={n_pos} neg={n_neg} score_min={s_min:.6f} score_max={s_max:.6f}", flush=True)
    
            # must have both classes
            if n_pos == 0 or n_neg == 0:
                print("[AUROC WARN] y_true_bin has only one class -> AUROC undefined", flush=True)
                auroc_ovr = float("nan")
            # if score is constant, AUROC is not meaningful
            elif s_min == s_max:
                print("[AUROC WARN] y_score_pos is constant -> AUROC undefined", flush=True)
                auroc_ovr = float("nan")
            else:
                try:
                    auroc_ovr = float(roc_auc_score(y_true_bin, y_score_pos))
                except Exception as e:
                    print(f"[AUROC ERR] roc_auc_score failed: {type(e).__name__}: {e}", flush=True)
                    auroc_ovr = float("nan")


    return {
        "accuarcy": acc,
        "precision": prec,
        "recall": rec,
        "sensitivity": sens,
        "f1_score": f1,
        "specificity": spec,
        "mcc": float(mcc),
        "kappa": float(kappa),
        "auroc_ovr": float(auroc_ovr),
    }

def _topk_accuracies_from_logits(logits: torch.Tensor, y: torch.Tensor, topk=(1,)):
    """
    logits: [B, C]
    y: [B]
    returns dict: {k: correct_count}
    """
    maxk = max(topk)
    # indices: [B, maxk]
    _, pred_topk = logits.topk(maxk, dim=1, largest=True, sorted=True)
    # [B, maxk] compare against y
    correct = pred_topk.eq(y.view(-1, 1))
    out = {}
    for k in topk:
        out[k] = int(correct[:, :k].any(dim=1).sum().item())
    return out


def _compute_ece(probs, true, n_bins=15, return_bins=False):
    import numpy as np

    probs = np.asarray(probs)
    true  = np.asarray(true)

    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accs  = (preds == true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confs, bin_edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_accs = np.zeros(n_bins, dtype=np.float64)
    bin_confs = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        m = (bin_ids == b)
        cnt = int(m.sum())
        bin_counts[b] = cnt
        if cnt > 0:
            bin_accs[b] = float(accs[m].mean())
            bin_confs[b] = float(confs[m].mean())

    ece = float(np.sum((bin_counts / max(len(true), 1)) * np.abs(bin_accs - bin_confs)))

    if return_bins:
        return ece, bin_edges, bin_accs, bin_confs, bin_counts
    return ece, bin_edges



def topk_accuracy_from_probs(probs: np.ndarray, true: np.ndarray, ks=(1, 3)):
    """
    probs: [N, C] probabilities
    true:  [N] int labels
    returns dict: {"top1":..., "top3":...}
    """
    probs = np.asarray(probs)
    true = np.asarray(true).astype(int)
    N, C = probs.shape

    # argsort descending
    top = np.argsort(-probs, axis=1)  # [N, C]
    out = {}
    for k in ks:
        kk = min(int(k), C)
        hits = (top[:, :kk] == true[:, None]).any(axis=1)
        out[f"top{k}"] = float(hits.mean()) if N > 0 else float("nan")
    return out


def ece_from_probs(probs: np.ndarray, true: np.ndarray, n_bins: int = 15):
    """
    Expected Calibration Error (ECE) for multiclass.
    Uses confidence = max prob, prediction = argmax.
    Returns: (ece, mce, bin_stats_list)
    bin_stats: list of dict per bin: {"count","acc","conf","gap","lo","hi"}
    """
    probs = np.asarray(probs)
    true = np.asarray(true).astype(int)

    conf = probs.max(axis=1)                  # [N]
    pred = probs.argmax(axis=1)               # [N]
    correct = (pred == true).astype(np.float32)

    # bins in [0,1]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    stats = []

    N = len(true)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        # include right edge in last bin
        if b == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        cnt = int(mask.sum())
        if cnt == 0:
            stats.append({"count": 0, "acc": float("nan"), "conf": float("nan"), "gap": float("nan"), "lo": float(lo), "hi": float(hi)})
            continue

        acc_b = float(correct[mask].mean())
        conf_b = float(conf[mask].mean())
        gap = abs(acc_b - conf_b)

        ece += (cnt / max(N, 1)) * gap
        mce = max(mce, gap)

        stats.append({"count": cnt, "acc": acc_b, "conf": conf_b, "gap": gap, "lo": float(lo), "hi": float(hi)})

    return float(ece), float(mce), stats


def plot_confusion_matrix_png(
    cm: np.ndarray,
    class_names,
    out_path: str,
    normalize: str | None = "true",   # "true" | "pred" | "all" | None
    title: str = "Confusion Matrix",
):
    """
    Saves a nice confusion matrix plot to out_path.
    normalize:
      - "true": row-normalized
      - "pred": col-normalized
      - "all":  global normalized
      - None: counts
    """
    cm = np.asarray(cm, dtype=np.float64)
    C = cm.shape[0]
    names = list(class_names) if class_names is not None else [f"c{i}" for i in range(C)]

    cm_plot = cm.copy()
    if normalize == "true":
        denom = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm, denom, out=np.zeros_like(cm), where=(denom != 0))
    elif normalize == "pred":
        denom = cm.sum(axis=0, keepdims=True)
        cm_plot = np.divide(cm, denom, out=np.zeros_like(cm), where=(denom != 0))
    elif normalize == "all":
        denom = cm.sum()
        cm_plot = cm / denom if denom != 0 else cm

    fig = plt.figure(figsize=(0.9 * C + 4, 0.9 * C + 3))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm_plot, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(C))
    ax.set_yticks(np.arange(C))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)

    # annotate
    thresh = cm_plot.max() * 0.5 if cm_plot.size else 0.0
    for i in range(C):
        for j in range(C):
            if normalize is None:
                txt = f"{int(cm[i, j])}"
            else:
                txt = f"{cm_plot[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color=("white" if cm_plot[i, j] > thresh else "black"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_diagram_png(bin_stats, out_path: str, title: str = "Reliability Diagram"):
    """
    bin_stats: output of ece_from_probs(...)[2]
    Plots accuracy vs confidence.
    """
    # filter bins with count>0
    xs, ys = [], []
    for b in bin_stats:
        if b["count"] and not (math.isnan(b["acc"]) or math.isnan(b["conf"])):
            xs.append(b["conf"])
            ys.append(b["acc"])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])     # perfect calibration line
    if len(xs) > 0:
        ax.scatter(xs, ys)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Optional text metrics (safe: code will skip if not installed / no texts)
try:
    import evaluate
except Exception:
    evaluate = None

try:
    from bert_score import score as bertscore
except Exception:
    bertscore = None


def safe_write_json(path: str, obj) -> None:
    """Best-effort JSON writer."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
    except OSError as e:
        print(f"[WARN] Could not write {path}: {e}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    roc_auc_score,
)
from tqdm import tqdm

def _jsonify_metrics(d):
    out = {}
    for k, v in d.items():
        # numpy scalar / torch scalar ? python float
        if hasattr(v, "item"):
            try:
                out[k] = float(v.item())
                continue
            except Exception:
                pass

        # list/tuple (e.g., per-class metrics) ? list of floats when possible
        if isinstance(v, (list, tuple)):
            vv = []
            for x in v:
                if hasattr(x, "item"):
                    x = x.item()
                try:
                    vv.append(float(x))
                except Exception:
                    vv.append(x)
            out[k] = vv
            continue

        # dict (nested) ? recurse
        if isinstance(v, dict):
            out[k] = _jsonify_metrics(v)
            continue

        # normal scalar
        try:
            out[k] = float(v)
        except Exception:
            out[k] = v
    return out

def _as_scalar(x):
    # torch / numpy scalar
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass

    # list/tuple -> summarize (mean). You can change to max if you want.
    if isinstance(x, (list, tuple)):
        xs = []
        for t in x:
            if hasattr(t, "item"):
                t = t.item()
            try:
                xs.append(float(t))
            except Exception:
                pass
        return float(sum(xs) / len(xs)) if xs else float("nan")

    # normal scalar
    try:
        return float(x)
    except Exception:
        return float("nan")


# ------------------------
# Utilities
# ------------------------
def get_trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return only trainable parameters (dramatically smaller checkpoints)."""
    trainable = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable[name] = p.detach().cpu()
    return trainable


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class FocalLoss(nn.Module):
    """Multiclass focal loss for logits."""
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, target, weight=self.weight, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def build_loss(cfg: dict, num_classes: int, device: torch.device) -> nn.Module:
    """
    cfg example:
      train:
        loss:
          name: ce          # ce | focal
          label_smoothing: 0.0
          focal_gamma: 2.0
        class_weights: balanced   # null | balanced | [..C..]
    """
    train_cfg = cfg.get("train", {})
    loss_cfg = train_cfg.get("loss", {}) or {}
    name = (loss_cfg.get("name", "ce") or "ce").lower()

    cw = train_cfg.get("class_weights", None)
    weight = None
    if isinstance(cw, (list, tuple)) and len(cw) == num_classes:
        weight = torch.tensor(cw, dtype=torch.float32, device=device)

    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0) or 0.0)

    if name in ("ce", "cross_entropy", "cross-entropy"):
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    if name in ("focal", "focal_loss"):
        gamma = float(loss_cfg.get("focal_gamma", 2.0) or 2.0)
        return FocalLoss(gamma=gamma, weight=weight, label_smoothing=label_smoothing)

    raise ValueError(f"Unknown loss.name: {name}")


# ------------------------
# Evaluation
# ------------------------

# Optional text metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

def _rouge_l_f1(pred: str, ref: str) -> float:
    """
    Simple ROUGE-L F1 based on LCS.
    No external deps.
    """
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    # LCS DP
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == pred_tokens[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    prec = lcs / max(n, 1)
    rec = lcs / max(m, 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _get_first(batch: dict, keys, default=None):
    for k in keys:
        if k in batch and batch[k] is not None:
            return batch[k]
    return default


def _resolve_amp_dtype(amp_dtype):
    # Accept torch.dtype directly
    if isinstance(amp_dtype, torch.dtype):
        return amp_dtype

    # Accept strings
    if isinstance(amp_dtype, str):
        s = amp_dtype.strip().lower()
        if s in ("fp16", "float16", "half", "16"):
            return torch.float16
        if s in ("bf16", "bfloat16"):
            return torch.bfloat16
        if s in ("fp32", "float32", "32"):
            return torch.float32  # usually not useful for autocast, but safe
    # Fallback
    return torch.float16


@torch.no_grad()
def evaluate_multiclass(
    model,
    val_loader,
    device="cuda",
    amp=False,
    amp_dtype=torch.float16,
    criterion=None,
    print_cls_report=False,

    class_names=None,
    cls_report_digits=4,

    # We may pass these, but we trust probs.shape[1] more
    num_classes=None,
    labels=None,
    label_names=None,

    # Debug
    debug_first_batch=False,
    debug_nan_inf=False,
    verbose_cm=True,

    # Binary TP/TN/FP/FN from multiclass (pick one class as "positive")
    positive_class=0,

    # Step/plots
    epoch=None,
    global_step=None,
    plot_reliability=True,

    # Top-k
    compute_topk=True,
    topk_list=(1, 3, 5),

    # Calibration (ECE)
    compute_calibration=True,
    ece_bins=15,

    # Plot/save
    save_dir=None,
    save_prefix="val",
    plot_cm=True,
    plot_roc=True,
    cm_normalize="true",
    run_name: str = None,
    model_name: str = None,
):

    import math
    import torch
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        f1_score, precision_score, recall_score,
        confusion_matrix, log_loss, roc_auc_score,
        matthews_corrcoef, cohen_kappa_score
    )
    

    def _sanitize_name(x: str) -> str:
        x = str(x) if x else "model"
        x = x.strip()
        x = re.sub(r"\s+", "_", x)
        x = re.sub(r"[^A-Za-z0-9._-]+", "_", x)
        return x[:150]

    
    def _plot_reliability(bin_edges, bin_accs, bin_confs, bin_counts, title=None):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    
        bin_edges = np.asarray(bin_edges, dtype=np.float64)
        bin_accs  = np.asarray(bin_accs, dtype=np.float64)
        bin_confs = np.asarray(bin_confs, dtype=np.float64)
    
        mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot([0, 1], [0, 1])  # perfect calibration line
        ax.plot(mids, bin_accs, marker="o", label="Accuracy per bin")
        ax.plot(mids, bin_confs, marker="s", label="Confidence per bin")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        if title:
            ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        return fig

    # --- helpers ---
    def _resolve_amp_dtype_local(dt):
        if isinstance(dt, torch.dtype):
            return dt
        if isinstance(dt, str):
            s = dt.strip().lower()
            if s in ("fp16", "float16", "half", "16"):
                return torch.float16
            if s in ("bf16", "bfloat16"):
                return torch.bfloat16
            if s in ("fp32", "float32", "32"):
                return torch.float32
        return torch.float16

    
    def _safe_div(a, b):
        return float(a) / float(b) if float(b) != 0.0 else 0.0
        

    def macro_average_dicts(dict_list, keys):
        out = {}
        for k in keys:
            vals = [d.get(k, float("nan")) for d in dict_list]
            out[k] = float(np.nanmean(vals))
        return out

    def _topk_acc(logits, y, k):
        # logits: [B,C], y: [B]
        k = int(k)
        k = min(k, logits.shape[1])
        topk = torch.topk(logits, k=k, dim=1).indices  # [B,k]
        correct = (topk == y.view(-1, 1)).any(dim=1).float().mean().item()
        return float(correct)

    def _ece(probs_np, true_np, n_bins=15):
        # probs_np: [N,C], true_np: [N]
        probs = probs_np.astype(np.float64)
        true = true_np.astype(np.int64)

        conf = probs.max(axis=1)             # [N]
        pred = probs.argmax(axis=1)          # [N]
        acc  = (pred == true).astype(np.float64)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        bin_accs = []
        bin_confs = []
        bin_counts = []
        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            if b == n_bins - 1:
                m = (conf >= lo) & (conf <= hi)
            else:
                m = (conf >= lo) & (conf < hi)

            cnt = int(m.sum())
            bin_counts.append(cnt)
            if cnt == 0:
                bin_accs.append(0.0)
                bin_confs.append(0.0)
                continue

            a = float(acc[m].mean())
            c = float(conf[m].mean())
            bin_accs.append(a)
            bin_confs.append(c)
            ece += (cnt / len(true)) * abs(a - c)

        return float(ece), bins, bin_accs, bin_confs, bin_counts

    
    def _plot_confusion_matrix(cm, class_names, normalize=None, title: str = "Confusion Matrix"):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")  # headless compute node safe
        import matplotlib.pyplot as plt
    
        cm_plot = cm.astype(np.float64)
    
        if normalize == "true":
            cm_plot = cm_plot / np.maximum(cm_plot.sum(axis=1, keepdims=True), 1e-12)
        elif normalize == "pred":
            cm_plot = cm_plot / np.maximum(cm_plot.sum(axis=0, keepdims=True), 1e-12)
        elif normalize == "all":
            cm_plot = cm_plot / max(cm_plot.sum(), 1e-12)
    
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_plot, interpolation="nearest")
        fig.colorbar(im, ax=ax)
    
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="Target Classes",
            xlabel="Predected Classes",
        )
        ax.tick_params(axis="x", rotation=45)
    
        if title:
            ax.set_title(title)
    
        # annotate
        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                txt = f"{cm_plot[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
                ax.text(j, i, txt, ha="center", va="center")
    
        fig.tight_layout()
        print ("plot Confusion matrix done", flush=True)
        return fig
        
    def _plot_multiclass_roc_ovr(y_true: np.ndarray,y_prob: np.ndarray, class_names=None,title: str = "ROC Curve", ):
        """
        Multiclass ROC curve (One-vs-Rest).
        Plots per-class ROC + micro-average + macro-average.
    
        y_true: (N,) int labels
        y_prob: (N,K) probabilities (softmax)
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve, auc
        except Exception as e:
            print(f"[EVAL WARN] ROC plot skipped (missing deps): {e}", flush=True)
            return None
    
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob, dtype=np.float64)
        N, K = y_prob.shape
    
        # binarize labels
        y_bin = np.zeros((N, K), dtype=np.int32)
        y_bin[np.arange(N), y_true] = 1
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        # Per-class ROC
        fpr = {}
        tpr = {}
        roc_auc = {}
    
        valid_classes = 0
        for k in range(K):
            # need both positive and negative samples
            if y_bin[:, k].min() == y_bin[:, k].max():
                roc_auc[k] = float("nan")
                continue
    
            fpr[k], tpr[k], _ = roc_curve(y_bin[:, k], y_prob[:, k])
            roc_auc[k] = auc(fpr[k], tpr[k])
            valid_classes += 1
    
            name = class_names[k] if (class_names is not None and k < len(class_names)) else f"class {k}"
            ax.plot(fpr[k], tpr[k], label=f"{name} (AUC={roc_auc[k]:.3f})")
    
        # Micro-average ROC (flatten all classes)
        try:
            fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
            auc_micro = auc(fpr_micro, tpr_micro)
            ax.plot(fpr_micro, tpr_micro, linestyle="--", label=f" Average (AUC={auc_micro:.3f})")
        except Exception:
            auc_micro = float("nan")
    
        # Macro-average ROC (mean TPR over common FPR grid)
#        try:
#            all_fpr = np.unique(np.concatenate([fpr[k] for k in range(K) if k in fpr]))
#            mean_tpr = np.zeros_like(all_fpr)
#            used = 0
#            for k in range(K):
#                if k not in fpr:
#                    continue
#                mean_tpr += np.interp(all_fpr, fpr[k], tpr[k])
#                used += 1
#            if used > 0:
#                mean_tpr /= used
#                auc_macro = auc(all_fpr, mean_tpr)
#                #ax.plot(all_fpr, mean_tpr, linestyle=":", label=f"macro (AUC={auc_macro:.3f})")
#            else:
#                auc_macro = float("nan")
#        except Exception:
#            auc_macro = float("nan")
    
        # Diagonal
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_title(title)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower right", fontsize="small")
    
        fig.tight_layout()
        print ("plot ROC Curve done", flush=True)
        roc_info = {
            "fpr_micro": fpr_micro,
            "tpr_micro": tpr_micro,
            "auc_micro": auc_micro,
        }
        return fig, roc_info

        
    
    
    def safe_div(num: float, den: float, default: float = 0.0) -> float:
        return num / den if den != 0 else default
    
    def binary_metrics_from_counts(tp: float, tn: float, fp: float, fn: float) -> Dict[str, float]:
        total = tp + tn + fp + fn
        acc = safe_div(tp + tn, total)
    
        prec = safe_div(tp, tp + fp)                 # PPV
        rec  = safe_div(tp, tp + fn)                 # TPR / Sensitivity
        f1   = safe_div(2 * prec * rec, prec + rec)
    
        fpr  = safe_div(fp, fp + tn)
        tnr  = safe_div(tn, tn + fp)                 # Specificity
        npv  = safe_div(tn, tn + fn)
    
        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        mcc = safe_div(tp * tn - fp * fn, math.sqrt(denom)) if denom > 0 else 0.0
    
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tpr": rec,
            "fpr": fpr,
            "tnr": tnr,
            "specificity": tnr,
            "ppv": prec,
            "npv": npv,
            "mcc": mcc,
        }
    
    def _multiclass_kappa_from_cm(cm: np.ndarray) -> float:
        """
        Cohen's kappa for multiclass from confusion matrix.
        kappa = (po - pe) / (1 - pe)
        po = observed agreement = trace/N
        pe = expected agreement = sum_i (row_i/N)*(col_i/N)
        """
        cm = np.asarray(cm, dtype=np.float64)
        N = cm.sum()
        if N <= 0:
            return 0.0
        po = np.trace(cm) / N
        row = cm.sum(axis=1) / N
        col = cm.sum(axis=0) / N
        pe = float(np.sum(row * col))
        return safe_div(po - pe, 1.0 - pe, default=0.0)
    
    
    
    def format_per_class_table(per_class_table, *, float_fmt="{:.4f}") -> str:
        """
        per_class_table: list[dict] with keys:
          class, support, precision, recall, f1, specificity
        Returns a pretty fixed-width table string.
        """
        if not per_class_table:
            return "(empty per-class table)"
    
        headers = ["class", "support", "precision", "recall", "f1", "specificity"]
    
        # Build rows as strings
        rows = []
        for r in per_class_table:
            rows.append([
                str(r.get("class", "")),
                str(int(round(float(r.get("support", 0.0))))),  # support as int
                float_fmt.format(float(r.get("precision", 0.0))),
                float_fmt.format(float(r.get("recall", 0.0))),
                float_fmt.format(float(r.get("f1", 0.0))),
                float_fmt.format(float(r.get("specificity", 0.0))),
            ])
    
        # Column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for j, cell in enumerate(row):
                col_widths[j] = max(col_widths[j], len(cell))
    
        def fmt_row(cells):
            return " | ".join(cells[j].ljust(col_widths[j]) for j in range(len(headers)))
    
        sep = "-+-".join("-" * w for w in col_widths)
    
        out = []
        out.append(fmt_row(headers))
        out.append(sep)
        for row in rows:
            out.append(fmt_row(row))
    
        return "\n".join(out)

    
    
    def _multiclass_auroc_ovr_macro(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        labels: Optional[Sequence[int]] = None,
    ) -> float:
        """
        Prob-based multiclass AUROC using OVR macro.
        Requires sklearn at runtime; returns nan if not available or not computable.
        """
        try:
            from sklearn.metrics import roc_auc_score
        except Exception:
            return float("nan")
    
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob, dtype=np.float64)
    
        if y_true.ndim != 1:
            y_true = y_true.reshape(-1)
    
        K = y_prob.shape[1]
        if labels is None:
            labels = list(range(K))
    
        # If a class is missing in y_true, roc_auc_score can fail for that class.
        # We'll compute per-class and nanmean.
        aucs = []
        for k in range(K):
            y_bin = (y_true == labels[k]).astype(np.int32)
            # Need both positives and negatives
            if y_bin.min() == y_bin.max():
                aucs.append(float("nan"))
                continue
            try:
                aucs.append(float(roc_auc_score(y_bin, y_prob[:, k])))
            except Exception:
                aucs.append(float("nan"))
    
        if np.all(np.isnan(aucs)):
            return float("nan")
        return float(np.nanmean(aucs))
    
    def multiclass_ovr_metrics(
        conf_mat: np.ndarray,
        *,
        y_true: Optional[Union[np.ndarray, Sequence[int]]] = None,
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Upgraded one-vs-rest metrics per class + macro/weighted averages
        + per-class table + multiclass AUROC (prob-based) + multiclass kappa (label-based).
    
        Args:
          conf_mat: (K,K) confusion matrix, rows=true, cols=pred
          y_true:   optional 1D true labels (needed for AUROC + kappa if you prefer label-based;
                    kappa is computed from conf_mat regardless)
          y_prob:   optional (N,K) probability matrix (needed for AUROC)
          class_names: optional list of names length K
    
        Returns dict:
          - overall_accuracy
          - kappa_multiclass
          - auroc_ovr_macro (nan if y_prob or sklearn unavailable)
          - per_class (dict keyed by class index)
          - per_class_table (list of rows: dicts with class, support, precision, recall, f1, specificity)
          - macro, weighted
          - support_total
        """
        cm = np.asarray(conf_mat, dtype=np.int64)
        assert cm.ndim == 2 and cm.shape[0] == cm.shape[1], "conf_mat must be square"
    
        K = cm.shape[0]
        N = float(cm.sum())
    
        row_sum = cm.sum(axis=1).astype(np.float64)   # support per class (true counts)
        col_sum = cm.sum(axis=0).astype(np.float64)
    
        # Per-class OvR metrics
        per_class: Dict[int, Dict[str, float]] = {}
        for k in range(K):
            tp = float(cm[k, k])
            fn = float(row_sum[k] - cm[k, k])
            fp = float(col_sum[k] - cm[k, k])
            tn = float(N - tp - fn - fp)
            d = binary_metrics_from_counts(tp, tn, fp, fn)
            d["support"] = float(row_sum[k])
            per_class[k] = d
    
        # Global accuracy
        overall_accuracy = float(np.trace(cm) / N) if N > 0 else 0.0
    
        # Multiclass kappa (label-based, computed from cm)
        kappa_multiclass = _multiclass_kappa_from_cm(cm)
    
        # Macro + weighted averages
        keys = ["precision", "recall", "f1", "tpr", "fpr", "tnr", "specificity", "ppv", "npv", "mcc"]
        macro = {k: float(np.mean([per_class[i][k] for i in range(K)])) for k in keys}
    
        weights = row_sum
        wsum = float(weights.sum()) if float(weights.sum()) > 0 else 1.0
        weighted = {
            k: float(np.sum([per_class[i][k] * weights[i] for i in range(K)]) / wsum)
            for k in keys
        }
    
        # Per-class table for publication/logging
        per_class_table = []
        for k in range(K):
            name = class_names[k] if (class_names is not None and k < len(class_names)) else str(k)
            per_class_table.append({
                "class": name,
                "support": per_class[k]["support"],
                "precision": per_class[k]["precision"],
                "recall": per_class[k]["recall"],
                "f1": per_class[k]["f1"],
                "specificity": per_class[k]["specificity"],
            })
        table_str = format_per_class_table(per_class_table)
        print(table_str, flush=True)
    
        # Multiclass AUROC (prob-based)
        auroc_ovr_macro = float("nan")
        if y_prob is not None and y_true is not None:
            y_true_arr = np.asarray(y_true)
            y_prob_arr = np.asarray(y_prob)
            if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == K and y_true_arr.shape[0] == y_prob_arr.shape[0]:
                auroc_ovr_macro = _multiclass_auroc_ovr_macro(y_true_arr, y_prob_arr, labels=list(range(K)))
    
        return {
            "overall_accuracy": overall_accuracy,
            "kappa_multiclass": kappa_multiclass,
            "auroc_ovr_macro": auroc_ovr_macro,
            "per_class": per_class,
            "per_class_table": table_str,   # ready to print as a table
            "macro": macro,
            "weighted": weighted,
            "support_total": float(N),
        }


    # --- eval loop ---
    model.eval()
    metrics = {}
    amp_dtype = _resolve_amp_dtype_local(amp_dtype)

    all_logits = []
    all_probs = []
    all_true = []
    all_pred = []
    total_loss = 0.0
    total_n = 0
    # --- Top-k counters (init ONCE before loop) ---
    topk_list = tuple(sorted(set(topk_list)))
    topk_correct = {k: 0 for k in topk_list}


    for i, batch in enumerate(val_loader):
        if isinstance(batch, (tuple, list)):
            x, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            def _first_present(d, keys):
                for k in keys:
                    if k in d and d[k] is not None:
                        return d[k]
                return None
        
            x = _first_present(batch, ["image", "images", "x", "pixel_values"])
            y = _first_present(batch, ["label", "labels", "y", "target", "targets"])


        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        if x is None or y is None:
            raise KeyError(f"Batch missing x/y. Keys={list(batch.keys()) if isinstance(batch, dict) else 'tuple'}")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long().view(-1)

        if amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(x)
        else:
            out = model(x)

        logits = out[0] if isinstance(out, (tuple, list)) else out
        
        probs_t = torch.softmax(logits.float(), dim=-1)  # [B,C] in fp32

        # --- HARD CHECK / DEBUG ---
        if debug_nan_inf:
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"[EVAL DEBUG] NaN/Inf in logits at batch {i}", flush=True)
                print("logits min/max:", float(torch.nanmin(logits)), float(torch.nanmax(logits)), flush=True)
        
            if torch.isnan(probs_t).any() or torch.isinf(probs_t).any():
                print(f"[EVAL DEBUG] NaN/Inf in probs at batch {i}", flush=True)
                print("probs min/max:", float(torch.nanmin(probs_t)), float(torch.nanmax(probs_t)), flush=True)
        
        


        # Ensure [B,C]
        if logits.dim() > 2:
            logits = logits.mean(dim=1)
            
        # --- Top-k accumulation ---
        if compute_topk:
            # logits: [B, C], y: [B]
            maxk = max(topk_list)
            _, topk_idx = torch.topk(logits, k=maxk, dim=1)   # [B, maxk]
            y_ = y.view(-1, 1)                                # [B, 1]
            matches = topk_idx.eq(y_)                         # [B, maxk] boolean
        
            for k in topk_list:
                # correct if true label appears in first k predictions
                topk_correct[k] += int(matches[:, :k].any(dim=1).sum().item())


        if debug_first_batch and i == 0:
            print(
                f"[EVAL DEBUG] logits.shape={tuple(logits.shape)} logits.dtype={logits.dtype} "
                f"y.shape={tuple(y.shape)} y.dtype={y.dtype} y.min={int(y.min())} y.max={int(y.max())}",
                flush=True
            )

        if debug_nan_inf:
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"[EVAL DEBUG] NaN/Inf in logits at batch {i}", flush=True)
                print("logits min/max:", float(logits.min()), float(logits.max()), flush=True)

#        probs = torch.softmax(logits.float(), dim=-1)
#        pred = torch.argmax(probs, dim=-1)
        # --- SANITIZE probs (prevents sklearn crash) ---
        # Convert to numpy for later, but sanitize in torch first
        eps = 1e-12
        probs_t = torch.nan_to_num(probs_t, nan=0.0, posinf=1.0, neginf=0.0)
        probs_t = torch.clamp(probs_t, eps, 1.0 - eps)
        probs_t = probs_t / probs_t.sum(dim=1, keepdim=True).clamp_min(eps)
        pred = torch.argmax(probs_t, dim=-1)

        if debug_nan_inf:
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print(f"[EVAL DEBUG] NaN/Inf in probs at batch {i}", flush=True)
                print("probs min/max:", float(probs.min()), float(probs.max()), flush=True)

        bs = y.size(0)
        total_n += bs
        
        if criterion is not None:
            loss = criterion(logits.float(), y)
            total_loss += float(loss.item()) * bs


        
        all_probs.append(probs_t.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        all_logits.append(logits.detach().cpu())
        #all_probs.append(probs.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())
        #all_pred.append(pred.detach().cpu().numpy())
        

    logits_all = torch.cat(all_logits, dim=0)              # [N,C]
    probs = np.concatenate(all_probs, axis=0)              # [N,C]
    true = np.concatenate(all_true, axis=0).astype(int)    # [N]
    pred = np.concatenate(all_pred, axis=0).astype(int)    # [N]
    
    if save_dir is not None and (save_prefix or "").lower().startswith("test"):
        print("DEBUG: save True and Pred label", flush=True)
        os.makedirs(save_dir, exist_ok=True)
        e_tag = f"e{int(epoch):04d}" if epoch is not None else "eXXXX"
        s_tag = f"s{int(global_step):06d}" if global_step is not None else "sXXXXXX"
        np.savez_compressed(
            os.path.join(save_dir, f"{save_prefix}_preds_{e_tag}_{s_tag}.npz"),
            true=true.astype(np.int32),
            pred=pred.astype(np.int32),
            probs=probs.astype(np.float32),
        )
    
    if compute_topk:
        for k in topk_list:
            metrics[f"top{k}_correct"] = float(topk_correct[k])  # count
            metrics[f"acc_top{k}"] = float(topk_correct[k] / max(total_n, 1))
        print("[DEBUG topk] total_n=", total_n, "topk_correct=", topk_correct, flush=True)

        # -------------------------
    # Classification report
    # -------------------------
    try:
        C = int(probs.shape[1])
        # choose names (safe)
        if class_names is None:
            target_names = [f"c{i}" for i in range(C)]
        else:
            # force list[str]
            target_names = [str(x) for x in class_names]
            if len(target_names) != C:
                print(f"[EVAL WARN] class_names length={len(target_names)} but C={C}. Using fallback names.")
                target_names = [f"c{i}" for i in range(C)]
    
        cls_rep = classification_report(
            true,
            pred,
            labels=list(range(C)),
            target_names=class_names,
            digits=int(cls_report_digits),
            zero_division=0,
        )
    
        if print_cls_report:
            print("\nclassification_report:\n" + cls_rep, flush=True)
    
        # also store it (useful for logs/json)
        metrics["classification_report"] = cls_rep
    
    except Exception as e:
        print(f"[EVAL WARN] could not compute classification_report: {e}", flush=True)


    C = probs.shape[1]
    if num_classes is not None and int(num_classes) != C:
        print(f"[EVAL WARN] num_classes passed={num_classes} but probs.shape[1]={C}. Using C={C}.", flush=True)

    labels_eff = list(range(C))  # always consistent
    if labels is None:
        labels = labels_eff

    # class names
    if label_names is None or len(label_names) < C:
        names = [f"c{k}" for k in range(C)]
    else:
        names = [str(label_names[k]) for k in range(C)]

    # confusion matrix
    cm = confusion_matrix(true, pred, labels=labels_eff)
    metrics_bin = multiclass_ovr_metrics(cm)

    if verbose_cm:
        print("confusion matrix:\n", cm, flush=True)
        print("[Manual metrics new claculation]", flush=True)
        # Print macro/weighted + accuracy
        print("overall_accuracy:", metrics_bin["overall_accuracy"])
        print("macro:", {k: round(v, 6) for k, v in metrics_bin["macro"].items() if k in ["precision","recall","f1","specificity","tpr", "fpr", "npv","mcc"]})
        print("weighted:", {k: round(v, 6) for k, v in metrics_bin["weighted"].items() if k in ["precision","recall","f1","specificity","tpr", "fpr", "npv","mcc"]})
        
        # (A) store the full structure
        metrics["ovr_metrics"] = metrics_bin
#        
#        # (B) flatten the main scalars (nice for W&B / CSV / json)
#        metrics["overall_accuracy"]  = float(metrics_bin["overall_accuracy"])
#        metrics["kappa_multiclass"]  = float(metrics_bin["kappa_multiclass"])
#        metrics["auroc_ovr_macro"]   = float(metrics_bin["auroc_ovr_macro"])
#        
#        # macro averages -> metrics["precision_ovr_macro"], etc.
#        for k, v in metrics_bin["macro"].items():
#            metrics[f"{k}_ovr_macro"] = float(v)
#        
#        # weighted averages -> metrics["precision_ovr_weighted"], etc.
#        for k, v in metrics_bin["weighted"].items():
#            metrics[f"{k}_ovr_weighted"] = float(v)
#        
#        # optional: store table (easy to print later)
#        metrics["per_class_table"] = metrics_bin["per_class_table"]


    # TP/TN/FP/FN for one-vs-rest (optional)
    
    # --- macro one-vs-rest metrics over ALL classes ---
    per_class_bin = []
    C = cm.shape[0]
    
    for k in range(C):
        tp = int(cm[k, k])
        fp = int(cm[:, k].sum() - cm[k, k])
        fn = int(cm[k, :].sum() - cm[k, k])
        tn = int(cm.sum() - (tp + fp + fn))
    
        #d = binary_metrics_from_confusion(tp, tn, fp, fn)
        y_true_bin = (true == k).astype(np.int32)
        y_score_pos = probs[:, k].astype(np.float64)
        d = binary_metrics_from_confusion(tp, tn, fp, fn, y_true_bin=y_true_bin, y_score_pos=y_score_pos)

        # (optional) keep counts too
        d.update({"tp": tp, "tn": tn, "fp": fp, "fn": fn, "pos_class": k})
        per_class_bin.append(d)
    
    
    # keys to macro-average
    keys = [
        "precision",
        "recall",
        "sensitivity",   # if you added it; otherwise remove
        "specificity",
        "f1_score",
        "mcc",           # if you added it; otherwise remove
        "kappa_",         # if you added it; otherwise remove
        "auroc_ovr",            # will be nan unless you pass scores; otherwise remove
    ]
    
    # only keep keys that exist in your dicts (avoids KeyError)
    keys = [k for k in keys if k in per_class_bin[0]]
    
    macro_bin = macro_average_dicts(per_class_bin, keys)
    
    # store results
    #metrics["ovr_per_class"] = per_class_bin              # list of dicts
    for k, v in macro_bin.items():
        metrics[f"{k}_macro"] = v                         # e.g., f1_bin_manual_macro
    
    if verbose_cm:
        print("[MANUAL OVR MACRO]", macro_bin, flush=True)


    
    
    # ---- Compare manual binary metrics to sklearn (same positive class) ----
    print("confusion matrix shape:", cm.shape[0], flush=True)
    print("Positve class",int(positive_class))
    if positive_class is not None:
        pos = int(positive_class)
        y_true_bin = (true == pos).astype(np.int32)
        y_pred_bin = (pred == pos).astype(np.int32)
    
        # sklearn binary metrics (sanity check)
        acc  = float((y_true_bin == y_pred_bin).mean())
        prec = _safe_div((y_true_bin & y_pred_bin).sum(), (y_pred_bin.sum()))
        rec  = _safe_div((y_true_bin & y_pred_bin).sum(), (y_true_bin.sum()))
        spec = _safe_div(((1 - y_true_bin) & (1 - y_pred_bin)).sum(), ((1 - y_true_bin).sum()))
        f1   = _safe_div(2 * prec * rec, prec + rec)
        tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
        tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
        fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
        fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
        sensitivity = _safe_div(tp, tp + fn)
        mcc_bin = float(matthews_corrcoef(y_true_bin, y_pred_bin))
        kappa_bin = float(cohen_kappa_score(y_true_bin, y_pred_bin))
    
        # AUROC (one-vs-rest) needs scores/probabilities for the positive class
        # You MUST provide a 1D score array aligned with `true/pred` order.
        # If you already have `probs_all` with shape [N, C] in this function, use:
        try:
            #auroc_ovr_macro = float(roc_auc_score(y_true_bin, probs, multi_class="ovr", average="macro"))
            auroc = roc_auc_score(y_true_bin, probs[:, pos])
        except Exception:
            auroc = float("nan")
    
#        metrics["acc_bin_manual"] = acc
#        metrics["precision_bin_manual"] = prec
#        metrics["recall_bin_manual"] = rec
#        metrics["f1_bin_manual"] = f1
#        metrics["specificity_bin_manual"] = spec
#        metrics["sensitivity_bin_manual"] = sensitivity
#        metrics["mcc_bin_manual"] = mcc_bin
#        metrics["kappa_bin_manual"] = kappa_bin
#        metrics["auroc_ovr"] = auroc
        #print(f"[EVAL BIN CHECK] acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} spec={spec:.4f}", flush=True)
        print(
        f"[EVAL BIN CHECK] pos={pos} "
        f"accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1 score={f1:.4f} "
        f"specificity={spec:.4f} sensitivity={sensitivity:.4f} "
        f"mcc={mcc_bin:.4f} kappa={kappa_bin:.4f} auroc_ovr={auroc:.4f}",
        flush=True)
    
    
    # per-class specificity
    specs = []
    for k in range(C):
        tpk = cm[k, k]
        fpk = cm[:, k].sum() - tpk
        fnk = cm[k, :].sum() - tpk
        tnk = cm.sum() - (tpk + fpk + fnk)
        denom = (tnk + fpk)
        specs.append((tnk / denom) if denom > 0 else 0.0)
    specificity_macro = float(np.mean(specs))
    sensitivity_macro = float(recall_score(true, pred, average="macro", zero_division=0))

    # log_loss (safe)
    # probs is numpy [N,C]
    eps = 1e-12
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, eps, 1.0 - eps)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), eps, None)

    ll = float(log_loss(true, probs, labels=labels_eff))

    # AUROC OVR macro (safe)
    aurocs = []
    for k in range(C):
        y_bin = (true == k).astype(np.int32)
        if y_bin.min() == y_bin.max():
            aurocs.append(float("nan"))
        else:
            aurocs.append(float(roc_auc_score(y_bin, probs[:, k])))
    auroc_ovr_macro = float(np.nanmean(aurocs)) if len(aurocs) else float("nan")



    # ECE
    ece_val = None
    if compute_calibration:
        ece_val, bins, bin_accs, bin_confs, bin_counts = _ece(probs, true, n_bins=ece_bins)

    # Resolve effective classes from probs
    C = probs.shape[1]
    labels_eff = list(range(C))
    
    # Names
    if label_names is None or len(label_names) < C:
        class_names = [f"c{k}" for k in range(C)]
    else:
        class_names = [str(label_names[k]) for k in range(C)]
    
    # Confusion matrix with effective labels
    cm = confusion_matrix(true, pred, labels=labels_eff)
    
    # ---- Save plots directory ----
    # ------------------------------------------------------------
    # Plot/save helpers (safe)
    tag = (save_prefix or "val")
    is_test_phase = str(tag).lower().startswith("test")
    print("DEBUG save_dir=", save_dir, "plot_cm=", plot_cm, "plot_roc=", plot_roc, "plot_reliability=", plot_reliability, flush=True)
    if save_dir is None and is_test_phase:
        print("DEBUG save_dir is None",  flush=True)
        save_dir = os.path.join(os.getcwd(), "eval_plots")
    _title_parts = []
    if save_prefix:
        _title_parts.append(str(save_prefix))
    if epoch is not None:
        _title_parts.append(f"epoch {epoch}")
    if global_step is not None:
        _title_parts.append(f"step {global_step}")
    title = " | ".join(_title_parts) if _title_parts else "eval"
    
    # ---- Confusion matrix plot ----
    if is_test_phase and plot_cm and save_dir is not None:
        print(" start plot confusion matrix:", flush=True)
        os.makedirs(save_dir, exist_ok=True)
        tag = save_prefix or "val"
    
        # safe tags even if epoch/global_step is None
        e_tag = f"e{int(epoch):04d}" if epoch is not None else "eXXXX"
        s_tag = f"s{int(global_step):06d}" if global_step is not None else "sXXXXXX"
    
        cm_path = os.path.join(save_dir, f"{tag}_cm_{e_tag}_{s_tag}.png")
        cm_title = f"Confusion Matrix"
        fig = _plot_confusion_matrix(cm, names, normalize=cm_normalize, title=cm_title)
        if fig is not None:
            fig.savefig(cm_path, bbox_inches="tight", dpi=200)
            plt.close(fig)
            metrics["confusion_matrix_path"] = cm_path
        else:
            print(f"[EVAL WARN] _plot_confusion_matrix returned None; skip save: {cm_path}", flush=True)
    
    
    # ---- ROC curve plot (TEST ONLY) ----
    if is_test_phase and plot_roc and save_dir is not None:
        try:
            os.makedirs(save_dir, exist_ok=True)
    
            e_tag = f"e{int(epoch):04d}" if epoch is not None else "eXXXX"
            s_tag = f"s{int(global_step):06d}" if global_step is not None else "sXXXXXX"
            roc_path = os.path.join(save_dir, f"{tag}_roc_{e_tag}_{s_tag}.png")
    
            roc_title = f"ROC Curve"
            if epoch is not None:
                roc_title += f" | epoch {epoch}"
            if global_step is not None:
                roc_title += f" | step {global_step}"
    
            fig, roc_info  = _plot_multiclass_roc_ovr(true, probs, class_names=names, title=roc_title)
            if model_name is None:
                model_name = run_name
            if fig is not None:
                fig.savefig(roc_path, bbox_inches="tight", dpi=200)
                plt.close(fig)
                metrics["roc_curve_path"] = roc_path
                ROC_NPZ_ROOT = "/scratch/ali95/LC25000_rocplots2"
                os.makedirs(ROC_NPZ_ROOT, exist_ok=True)
                safe_name = _sanitize_name(model_name)   # MUST be the benchmark model name
                npz_path = os.path.join(ROC_NPZ_ROOT, f"{safe_name}_roc_micro.npz")
                print("[DEBUG ROC] model_name =", model_name, flush=True)

                
                np.savez(
                    npz_path,
                    fpr_micro=roc_info["fpr_micro"],
                    tpr_micro=roc_info["tpr_micro"],
                    auc_micro=roc_info["auc_micro"],
                )
                metrics["roc_micro_npz_path"] = npz_path
            else:
                print(f"[EVAL WARN] ROC figure None; skip save: {roc_path}", flush=True)
        except Exception as e:
            print(f"[EVAL WARN] ROC plot failed: {e}", flush=True)

    
    
    # ---- Calibration / ECE + reliability plot ----
    ece = None
    if compute_calibration:
        # Expect: ece (float), plus bin stats for plotting
        # Make sure your _compute_ece returns these 5 items (recommended).
        # If your current _compute_ece returns only (ece, bins), update it accordingly.
        try:
            ece, bin_edges, bin_accs, bin_confs, bin_counts = _compute_ece(
                probs, true, n_bins=ece_bins, return_bins=True
            )
        except TypeError:
            # fallback if your _compute_ece returns fewer values
            ece, _ = _compute_ece(probs, true, n_bins=ece_bins)
            bin_edges = np.linspace(0.0, 1.0, ece_bins + 1)
            bin_accs = None
            bin_confs = None
            bin_counts = None
    
        metrics["ece"] = float(ece) if ece is not None else float("nan")
    
        if is_test_phase and plot_reliability and save_dir is not None and (bin_accs is not None):
            os.makedirs(save_dir, exist_ok=True)
            tag = save_prefix or "val"
            e_tag = f"e{int(epoch):04d}" if epoch is not None else "eXXXX"
            s_tag = f"s{int(global_step):06d}" if global_step is not None else "sXXXXXX"
            rel_path = os.path.join(save_dir, f"{tag}_reliability_{e_tag}_{s_tag}.png")
    
            rel_title = f"{tag} Reliability (ECE={ece:.4f})"
            if epoch is not None:
                rel_title += f" | epoch {epoch}"
            if global_step is not None:
                rel_title += f" | step {global_step}"
    
            # IMPORTANT: call _plot_reliability with the signature it actually supports.
            # Recommended signature: _plot_reliability(bin_edges, bin_accs, bin_confs, bin_counts, title=None)
            fig = _plot_reliability(bin_edges, bin_accs, bin_confs, bin_counts, title=rel_title)
    
            if fig is not None:
                fig.savefig(rel_path, bbox_inches="tight", dpi=200)
                plt.close(fig)
                metrics["reliability_plot_path"] = rel_path
            else:
                print(f"[EVAL WARN] _plot_reliability returned None; skip save: {rel_path}", flush=True)

    
    # main metrics
    f1_per_class = f1_score(true, pred, average=None, zero_division=0)
    precision_per_class = precision_score(true, pred, average=None, zero_division=0)
    recall_per_class = recall_score(true, pred, average=None, zero_division=0)

    metrics.update({
        "acc": float(accuracy_score(true, pred)),
        "log_loss": ll,
        "auroc_ovr_macro": auroc_ovr_macro,
        "mcc": float(matthews_corrcoef(true, pred)),
        "kappa": float(cohen_kappa_score(true, pred)),
    
        })

    # ECE
    if compute_calibration and (ece is not None):
        metrics["ece"] = float(ece) 


    # add TP/TN/FP/FN if requested
    if positive_class is not None and tp is not None:
        metrics["tp"] = tp
        metrics["tn"] = tn
        metrics["fp"] = fp
        metrics["fn"] = fn

    # val_loss
    if criterion is not None:
        metrics["val_loss"] = float(total_loss / max(total_n, 1))

    # plots
    rel_path = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        tag = save_prefix or "val"
        #rel_path = os.path.join(save_dir, f"{tag}_reliability_e{int(epoch):04d}_s{int(global_step):06d}.png")
        e_tag = f"e{int(epoch):04d}" if epoch is not None else "eXXXX"
        s_tag = f"s{int(global_step):06d}" if global_step is not None else "sXXXXXX"
        rel_path = os.path.join(save_dir, f"{tag}_reliability_{e_tag}_{s_tag}.png")

    
    if plot_reliability and rel_path is not None:
        fig = _plot_reliability(bin_edges, bin_accs, bin_confs, bin_counts, title=title)
        if fig is not None:
            fig.savefig(rel_path, bbox_inches="tight", dpi=200)
            plt.close(fig)
            metrics["reliability_path"] = rel_path


    return metrics


# ------------------------
# Training
# ------------------------
def train_multiclass(model, train_loader, val_loader, cfg, run_dir, wandb=None):
    device = torch.device(cfg.get("device", "cuda"))
    model.to(device)

    train_cfg = cfg.get("train", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}

    epochs = int(train_cfg.get("epochs", 10))
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    amp = bool(train_cfg.get("amp", True))
    log_every = int(train_cfg.get("log_every", 50))
    grad_clip = float(train_cfg.get("grad_clip", 0.0) or 0.0)
    accum_steps = int(train_cfg.get("accum_steps", 1) or 1)

    # selection for best checkpoint
    select_metric = str(eval_cfg.get("select_metric", "f1_macro"))
    select_mode = str(eval_cfg.get("select_mode", "max")).lower()  # max|min
    report_metrics = eval_cfg.get("report_metrics", None)

    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    
    bad = [(n, p.dtype, p.shape) for n,p in model.named_parameters()
       if p.requires_grad and p.dtype == torch.float16]

    print("FP16 trainable params:", len(bad))
    for x in bad[:40]:
        print(x)
        
    # after requires_grad flags are set
    if hasattr(model, "backbone") and hasattr(model.backbone, "visual_encoder"):
        model.backbone.visual_encoder.float()
    
    # also often needed
    if hasattr(model.backbone, "ln_vision") and model.backbone.ln_vision is not None:
        model.backbone.ln_vision.float()
    
    # if you also train Qformer
    if hasattr(model.backbone, "Qformer") and model.backbone.Qformer is not None:
        model.backbone.Qformer.float()
    
    # your classification head should also be fp32
    if hasattr(model, "classifier"):
        model.classifier.float()


    # Optimizer over trainable parameters only
    params = [p for p in model.parameters() if p.requires_grad]
    opt_cfg = train_cfg.get("optimizer", {}) or {}
    opt_name = str(opt_cfg.get("name", "adamw")).lower()
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps = float(opt_cfg.get("eps", 1e-8))

    if opt_name in ("adamw", "adam"):
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    # Scheduler (cosine + warmup)
    sched_cfg = train_cfg.get("scheduler", {}) or {}
    sched_name = str(sched_cfg.get("name", "none")).lower()
    total_steps = max(1, epochs * max(1, len(train_loader)) // accum_steps)
    warmup_steps = int(sched_cfg.get("warmup_steps", 0) or 0)
    warmup_ratio = float(sched_cfg.get("warmup_ratio", 0.0) or 0.0)
    if warmup_steps <= 0 and warmup_ratio > 0:
        warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    min_lr = float(sched_cfg.get("min_lr", 0.0) or 0.0)

    scheduler = None
    if sched_name in ("cosine", "cosine_warmup", "cosine_with_warmup"):
        def lr_lambda(step: int):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            # cosine decay
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            min_factor = min_lr / lr if lr > 0 else 0.0
            return min_factor + (1.0 - min_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    elif sched_name in ("none", "", None):
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {sched_name}")

    #scaler = torch.cuda.amp.GradScaler(enabled=amp)
    use_amp = True
    #use_amp = bool(amp) and (device.startswith("cuda") or device == "cuda")
    amp_dtype = "bf16"  # or read from cfg
    
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == "fp16"))

    # loss
    num_classes = int(cfg["data"]["num_classes"])


    criterion = build_loss(cfg, num_classes=num_classes, device=device)
    print("LOSS OBJECT:", criterion)
    print("LOSS CLASS :", criterion.__class__)

    best_score = None
    best_path = os.path.join(run_dir, "best_trainable.pt")
    last_path = os.path.join(run_dir, "last_trainable.pt")

    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = []

        opt.zero_grad(set_to_none=True)

        for it, batch in enumerate(tqdm(train_loader, desc=f"train e{epoch}", leave=False), start=1):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            use_amp = bool(amp)
            dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
            #with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = criterion(logits.float(), y)
                loss = loss / float(accum_steps)

            scaler.scale(loss).backward()

            if (it % accum_steps) == 0:
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

                global_step += 1

                running.append(float(loss.detach().cpu()) * float(accum_steps))

                if log_every and (global_step % log_every == 0):
                    lr_now = opt.param_groups[0]["lr"]
                    print(f"[step {global_step}] train/loss={np.mean(running[-log_every:]):.4f} lr={lr_now:.3e}")
                    if wandb:
                        wandb.log(
                            {
                                "step": global_step,
                                "train/loss": float(np.mean(running[-log_every:])),
                                "lr": float(lr_now),
                            },
                            step=global_step,
                        )

        train_loss = float(np.mean(running)) if running else float("nan")

        # Validation (per-epoch)
        #val_metrics = evaluate_multiclass(model, val_loader, device=device, amp=amp, criterion=criterion)
        amp_dtype = cfg.get("train", {}).get("amp_dtype", "fp16") # or read from your sub_cfg dict
        #val_metrics = evaluate_multiclass(model, val_loader, device=device, amp=amp, criterion=criterion, amp_dtype=amp_dtype)
        data_cfg = cfg.get("data", {}) or {}
        plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        val_metrics = evaluate_multiclass(
            model,
            val_loader,
            device=device,
            amp=amp,
            amp_dtype=amp_dtype,
            criterion=criterion,
            positive_class=0,
            verbose_cm=True,
        
            num_classes=int(data_cfg.get("num_classes", 0)) or None,
            label_names=data_cfg.get("classes", None),   # <-- this is the fix
        
            epoch=epoch,
            global_step=global_step,
            save_dir=plots_dir,
            save_prefix="val",    # chose plot val or test
        
            compute_topk=True,
            topk_list=(1, 3),
        
            compute_calibration=True,
            ece_bins=15,
            plot_reliability=True,
        
            plot_cm=True,
            plot_roc=True,
            cm_normalize="true",
            print_cls_report=True,
        )
        



        #val_loss = float(val_metrics.get("loss", float("nan")))
        val_loss = float(val_metrics.get("val_loss", float("nan")))

        # choose best score
        score = float(val_metrics.get(select_metric, val_loss))
        is_best = False
        if best_score is None:
            is_best = True
        else:
            if select_mode == "min":
                is_best = score < best_score
            else:
                is_best = score > best_score

        if is_best:
            best_score = score
            torch.save({"trainable": get_trainable_state_dict(model), "epoch": epoch, "score": score}, best_path)

        # Always save last
        torch.save({"trainable": get_trainable_state_dict(model), "epoch": epoch, "score": score}, last_path)

        # Log metrics row
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": float(opt.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best": bool(is_best),
            "select_metric": select_metric,
            "score": float(score),
            "val_metrics": _jsonify_metrics(val_metrics),
        }

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        # write a compact "latest" file for quick parsing
        safe_write_json(os.path.join(run_dir, "last_metrics.json"), row)

        # Console summary (filter report_metrics if provided)
        if isinstance(report_metrics, (list, tuple)) and len(report_metrics) > 0:
            rep = {k: _as_scalar(val_metrics.get(k, float("nan"))) for k in report_metrics}
        else:
            rep = {k: float(v) for k, v in val_metrics.items()}

        #rep_str = " ".join([f"{k}={v:.4f}" for k, v in rep.items() if isinstance(v, (int, float)) and not np.isnan(v)])
        rep_str_parts = []
        for k, v in val_metrics.items():
            if isinstance(v, list):
                rep_str_parts.append(f"{k}={v}")
            elif isinstance(v, (float, int)):
                rep_str_parts.append(f"{k}={v:.4f}")
        rep_str = " ".join(rep_str_parts)

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} {select_metric}={score:.4f} " +
            ("(BEST) " if is_best else "") + rep_str
        )


        if wandb:
          wandb_payload = {}
          for k, v in val_metrics.items():
              # only log scalars
              if isinstance(v, (int, float, np.floating, np.integer)):
                  wandb_payload[f"val/{k}"] = float(v)
          wandb.log(
              {
                  "epoch": epoch,
                  "train/loss_epoch": float(train_loss),
                  "val/loss": float(val_loss),
                  **wandb_payload,
              },
              step=global_step,
          )
 
                    

    return {"best_path": best_path, "best_score": best_score}

