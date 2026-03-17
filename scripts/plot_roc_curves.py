import os, glob
import numpy as np
import matplotlib.pyplot as plt

def plot_many_micro_rocs(npz_files, out_png="roc_all_models.png", title="Averaged ROC curves across models (Kvasir)"):
    # Load all first (and label from filename)
    items = []
    for f in npz_files:
        base = os.path.basename(f)
        if base.startswith(".nfs"):
            continue
        if not base.endswith("_roc_micro.npz"):
            continue

        d = np.load(f)
        fpr = np.asarray(d["fpr_micro"], dtype=float)
        tpr = np.asarray(d["tpr_micro"], dtype=float)
        auc_micro = float(d["auc_micro"])

        label = base.replace("_roc_micro.npz", "")
        items.append((auc_micro, fpr, tpr, label, f))

    if not items:
        raise SystemExit("No valid *_roc_micro.npz files found.")

    # Sort by highest AUC first (curve + legend order)
    items.sort(key=lambda x: x[0], reverse=True)

    plt.figure()
    for auc_micro, fpr, tpr, lab, _ in items:
        plt.plot(fpr, tpr, label=f"{lab} (AUC={auc_micro:.2f})")  # show more digits

    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)

    # Put origin exactly at (0,0) and remove padding (BEFORE savefig)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.margins(x=0.0, y=0.0)
    plt.gca().set_xmargin(0.0)
    plt.gca().set_ymargin(0.0)

    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print("Saved:", out_png)

if __name__ == "__main__":
    ROC_DIR = "/scratch/ali95/kvasir_rocplots"
    npz_files = sorted(glob.glob(os.path.join(ROC_DIR, "*_roc_micro.npz")))
    plot_many_micro_rocs(npz_files, out_png=os.path.join(ROC_DIR, "roc_all_models.png"))
