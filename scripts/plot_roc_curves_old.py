import os, glob
import numpy as np
import matplotlib.pyplot as plt

import os, numpy as np
import matplotlib.pyplot as plt

def plot_many_micro_rocs(npz_files, labels=None, out_png="roc_9_models.png", title=None, auc_decimals=3):
    # Load all first
    items = []
    for i, f in enumerate(npz_files):
        d = np.load(f)
        fpr = d["fpr_micro"]
        tpr = d["tpr_micro"]
        auc_micro = float(d["auc_micro"])
        lab = labels[i] if labels is not None else os.path.splitext(os.path.basename(f))[0]
        items.append((auc_micro, fpr, tpr, lab))

    # Sort: highest AUC first
    items.sort(key=lambda x: x[0], reverse=True)

    plt.figure()
    for auc_micro, fpr, tpr, lab in items:
        plt.plot(fpr, tpr, label=f"{lab} (AUC={auc_micro:.{auc_decimals}f})")

    # Diagonal
    plt.plot([0, 1], [0, 1], "--", linewidth=1)

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title or "ROC Curves for all models")
    plt.legend(loc="lower right", fontsize="small")

    # Force origin + remove extra whitespace near (0,0)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.margins(x=0.0, y=0.0)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out_png)




#def plot_many_micro_rocs(npz_files, labels=None, out_png="roc_9_models.png", title=None):
#    if labels is None:
#        labels = [os.path.splitext(os.path.basename(f))[0] for f in npz_files]
#
#    # Load first
#    items = []
#    for f, lab in zip(npz_files, labels):
#        d = np.load(f)
#        fpr = np.asarray(d["fpr_micro"], dtype=float)
#        tpr = np.asarray(d["tpr_micro"], dtype=float)
#        auc_micro = float(d["auc_micro"])
#        items.append((auc_micro, fpr, tpr, lab))
#
#    # Sort by highest AUC first (affects curve + legend order)
#    items.sort(key=lambda x: x[0], reverse=True)
#
#    plt.figure()
#
#    for auc_micro, fpr, tpr, lab in items:
#        plt.plot(fpr, tpr, label=f"{lab} (AUC={auc_micro:.3f})")
#
#    plt.plot([0, 1], [0, 1], "--", linewidth=1)
#
#    plt.xlabel("False Positive Rate (FPR)")
#    plt.ylabel("True Positive Rate (TPR)")
#    plt.title(title or "ROC Curve for all models")
#
#    # IMPORTANT: set these BEFORE savefig
#    plt.xlim(0.0, 1.0)
#    plt.ylim(0.0, 1.05)
#
#    # Remove auto padding so (0,0) is exactly at the corner
#    plt.margins(x=0.0, y=0.0)
#    plt.gca().set_xmargin(0.0)
#    plt.gca().set_ymargin(0.0)
#    plt.gca().autoscale(enable=False)
#
#    plt.legend(loc="lower right", fontsize="small")
#    plt.tight_layout(pad=0.0)
#
#    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.0)
#    plt.close()
#    print("Saved:", out_png)
    
#def plot_many_micro_rocs(npz_files, labels=None, out_png="roc_9_models.png", title=None):
#    if labels is None:
#        labels = [os.path.splitext(os.path.basename(f))[0] for f in npz_files]
#
#    
#    items = []
#    for f, lab in zip(npz_files, labels):
#        d = np.load(f)
#        fpr = d["fpr_micro"]
#        tpr = d["tpr_micro"]
#        auc_micro = float(d["auc_micro"])
#        plt.plot(fpr, tpr, label=f"{lab} (AUC={auc_micro:.3f})")
#        items.append((auc_micro, fpr, tpr, lab))
#        
#    items.sort(key=lambda x: x[0], reverse=True)  # highest AUC first
#    
#    plt.figure()
#    for auc_micro, fpr, tpr, lab in items:
#        plt.plot(fpr, tpr, label=f"{lab} (AUC={auc_micro:.3f})")
#
#    plt.plot([0, 1], [0, 1], "--", linewidth=1)
#    plt.xlabel("False Positive Rate (FPR)")
#    plt.ylabel("True Positive Rate (TPR)")
#    plt.title(title or "ROC Curve for all models")
#    plt.legend(loc="lower right", fontsize="small")
#    plt.tight_layout()
#    plt.savefig(out_png, dpi=200)
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.margins(x=0.0, y=0.0)
#    plt.gca().set_xmargin(0.0)
#    plt.gca().set_ymargin(0.0)
#    plt.gca().autoscale(enable=False)
#    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.0)
#    plt.close()
#    print("Saved:", out_png)

#def plot_many_micro_rocs(npz_files, labels=None, out_png="roc_models.png", title=None):
#    if labels is None:
#        labels = [os.path.splitext(os.path.basename(f))[0] for f in npz_files]
#
#    # ---- load + sort by AUC (high -> low) ----
#    items = []
#    for f, lab in zip(npz_files, labels):
#        d = np.load(f)
#        fpr = d["fpr_micro"].astype(float)
#        tpr = d["tpr_micro"].astype(float)
#        auc_micro = float(d["auc_micro"])
#        items.append((auc_micro, fpr, tpr, lab))
#
#    items.sort(key=lambda x: x[0], reverse=True)  # highest AUC first
#    # -----------------------------------------
#
#    fig, ax = plt.subplots()
#
#    for auc_micro, fpr, tpr, lab in items:
#        ax.plot(fpr, tpr, label=f"{lab} (AUC={auc_micro:.3f})")
#
#    ax.plot([0, 1], [0, 1], "--", linewidth=1)
#
#    ax.set_xlabel("False Positive Rate (FPR)")
#    ax.set_ylabel("True Positive Rate (TPR)")
#    ax.set_title(title or "ROC Curve for all models")
#
#    ax.set_xlim(0.0, 1.0)
#    ax.set_ylim(0.0, 1.05)
#    ax.margins(x=0.0, y=0.0)
#
#    ax.legend(loc="lower right", fontsize="small")
#    fig.tight_layout()
#    fig.savefig(out_png, dpi=200, bbox_inches="tight")
#    plt.close(fig)
#    print("Saved:", out_png)


#def plot_many_micro_rocs(npz_files, labels=None, out_png="roc_9_models.png", title=None):
#    if labels is None:
#        labels = [os.path.splitext(os.path.basename(f))[0] for f in npz_files]
#
#    # Load + collect first
#    items = []
#    for f, lab in zip(npz_files, labels):
#        d = np.load(f)
#        fpr = np.asarray(d["fpr_micro"], dtype=float)
#        tpr = np.asarray(d["tpr_micro"], dtype=float)
#        auc_micro = float(d["auc_micro"])
#        items.append((auc_micro, fpr, tpr, lab))
#
#    # Sort by highest AUC first (draw order + legend order)
#    items.sort(key=lambda x: x[0], reverse=True)
#
#    fig, ax = plt.subplots()
#
#    for auc_micro, fpr, tpr, lab in items:
#        ax.plot(fpr, tpr, label=f"{lab} (AUC={auc_micro:.3f})")
#
#    ax.plot([0, 1], [0, 1], "--", linewidth=1)
#
#    ax.set_xlabel("False Positive Rate (FPR)")
#    ax.set_ylabel("True Positive Rate (TPR)")
#    ax.set_title(title or "ROC Curve for all models")
#
#    # ---- IMPORTANT: set these BEFORE savefig ----
#    ax.set_xlim(0.0, 1.0)
#    ax.set_ylim(0.0, 1.05)
#
#    # Remove auto padding so (0,0) is exactly at the corner
#    ax.margins(x=0.0, y=0.0)
#    ax.set_xmargin(0.0)
#    ax.set_ymargin(0.0)
#    ax.autoscale(enable=False)  # prevents matplotlib from re-expanding after margins
#
#    ax.legend(loc="lower right", fontsize="small")
#    fig.tight_layout(pad=0.0)
#
#    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.0)
#    plt.close(fig)
#    print("Saved:", out_png)


#def plot_many_micro_rocs(npz_files, labels=None, out_png="roc_9_models.png", title=None):
#    if labels is None:
#        labels = [os.path.splitext(os.path.basename(f))[0] for f in npz_files]
#
#    plt.figure()
#    for f, lab in zip(npz_files, labels):
#        d = np.load(f)
#        fpr = d["fpr_micro"]
#        tpr = d["tpr_micro"]
#        auc_micro = float(d["auc_micro"])
#        plt.plot(fpr, tpr, label=f"{lab} (AUC={auc_micro:.3f})")
#
#        
#    plt.plot([0, 1], [0, 1], "--", linewidth=1)
#    plt.xlabel("False Positive Rate (FPR)")
#    plt.ylabel("True Positive Rate (TPR)")
#    plt.title(title or "ROC Curve for all models")
#    plt.legend(loc="lower right", fontsize="small")
#    plt.tight_layout()
#    plt.savefig(out_png, dpi=200)
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.margins(x=0.0, y=0.0)
#    plt.gca().set_xmargin(0.0)
#    plt.gca().set_ymargin(0.0)
#    plt.gca().autoscale(enable=False)
##    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.0)
##    plt.close()
#    print("Saved:", out_png)

if __name__ == "__main__":
    roc_dir = "/nfs/speed-scratch/a_alguma/kvasir_rocplots"
    npz_files = sorted(glob.glob(os.path.join(roc_dir, "*.npz")))

    if not npz_files:
        raise SystemExit(f"No ROC .npz files found in {roc_dir}")

    # Optional: provide nicer labels (edit this part if needed)
    labels = [
        os.path.basename(f)
        .replace(".npz", "")
        .replace("_roc_micro", "")
        for f in npz_files
    ]

    out_png = os.path.join(roc_dir, "roc_all_models.png")
    plot_many_micro_rocs(npz_files, labels=labels, out_png=out_png)
    print(f"[OK] Saved: {out_png}")
