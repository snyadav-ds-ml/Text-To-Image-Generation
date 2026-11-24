"""
Generate all result plots for the report.
Run: python results/plots.py
"""
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# A01 — Baseline vs LoRA
# ---------------------------------------------------------------------------
def plot_A01():
    models    = ["SD v1.5\n(Baseline)", "SD v1.5\n+ LoRA"]
    fid       = [87.25, 85.49]
    kid       = [0.0357, 0.0310]
    clip      = [31.34, 31.38]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#4C72B0", "#DD8452"]

    for ax, vals, title, ylabel in zip(
        axes,
        [fid, kid, clip],
        ["FID (lower is better)", "KID (lower is better)", "CLIP Score (higher is better)"],
        ["FID", "KID", "CLIP Score"]
    ):
        bars = ax.bar(models, vals, color=colors, width=0.4)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=10)

    plt.suptitle("A01: Baseline vs LoRA Fine-tuned", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/A01_baseline_vs_lora.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved A01")

# ---------------------------------------------------------------------------
# A02 — Guidance Scale
# ---------------------------------------------------------------------------
def plot_A02():
    cfg    = [1, 3, 5, 7.5, 10, 15]
    fid    = [122.86, 80.68, 78.96, 85.49, 92.57, 113.04]
    kid    = [0.0599, 0.0336, 0.0294, 0.0308, 0.0335, 0.0444]
    clip   = [26.73, 31.04, 31.48, 31.38, 31.12, 30.38]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, vals, title, ylabel in zip(
        axes,
        [fid, kid, clip],
        ["FID (lower is better)", "KID (lower is better)", "CLIP Score (higher is better)"],
        ["FID", "KID", "CLIP Score"]
    ):
        ax.plot(cfg, vals, 'o-', color="#4C72B0", linewidth=2, markersize=6)
        ax.axvline(x=7.5, color='red', linestyle='--', alpha=0.5, label='default (7.5)')
        ax.set_xlabel("Guidance Scale")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    plt.suptitle("A02: Guidance Scale Ablation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/A02_guidance_scale.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved A02")

# ---------------------------------------------------------------------------
# A03 — Sampling Steps
# ---------------------------------------------------------------------------
def plot_A03():
    steps      = [10, 20]
    fid        = [83.03, 82.59]
    kid        = [0.0307, 0.0297]
    clip       = [31.42, 31.45]
    secs       = [4.53, 8.56]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, vals, title, ylabel in zip(
        axes,
        [fid, kid, clip, secs],
        ["FID", "KID", "CLIP Score", "Seconds/image"],
        ["FID", "KID", "CLIP Score", "Seconds"]
    ):
        ax.plot(steps, vals, 'o-', color="#4C72B0", linewidth=2, markersize=6)
        ax.set_xlabel("Sampling Steps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.suptitle("A03: Sampling Steps Ablation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/A03_sampling_steps.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved A03 (partial — steps 30/50/100 pending)")

# ---------------------------------------------------------------------------
# A04 — Scheduler
# ---------------------------------------------------------------------------
def plot_A04():
    schedulers = ["DDIM\n10", "DDIM\n30", "DPM++\n10", "Euler\n10", "Euler\n30"]
    fid        = [97.08, 80.46, 83.03, 87.30, 82.26]
    kid        = [0.0482, 0.0325, 0.0307, 0.0324, 0.0292]
    clip       = [31.69, 31.83, 31.42, 31.46, 31.55]
    secs       = [4.47, 13.16, 4.52, 4.56, 12.78]

    x = range(len(schedulers))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, vals, title, ylabel in zip(
        axes,
        [fid, kid, clip, secs],
        ["FID", "KID", "CLIP Score", "Seconds/image"],
        ["FID", "KID", "CLIP Score", "Seconds"]
    ):
        bars = ax.bar(x, vals, color="#4C72B0")
        ax.set_xticks(x)
        ax.set_xticklabels(schedulers, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.suptitle("A04: Scheduler Ablation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/A04_scheduler.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved A04 (partial — pndm pending)")

# ---------------------------------------------------------------------------
# A06 — Training Steps
# ---------------------------------------------------------------------------
def plot_A06():
    steps = [500, 1000, 2000]
    fid   = [79.89, 82.16, 85.49]
    kid   = [0.0312, 0.0304, 0.0308]
    clip  = [31.70, 31.53, 31.38]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, vals, title, ylabel in zip(
        axes,
        [fid, kid, clip],
        ["FID (lower is better)", "KID (lower is better)", "CLIP Score (higher is better)"],
        ["FID", "KID", "CLIP Score"]
    ):
        ax.plot(steps, vals, 'o-', color="#4C72B0", linewidth=2, markersize=6)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.suptitle("A06: Training Steps Ablation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/A06_training_steps.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved A06")

# ---------------------------------------------------------------------------
# A07 — Negative Prompt
# ---------------------------------------------------------------------------
def plot_A07():
    models = ["No Negative\nPrompt", "With Negative\nPrompt"]
    fid    = [85.49, 86.06]
    kid    = [0.0308, 0.0325]
    clip   = [31.38, 30.94]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#4C72B0", "#DD8452"]
    for ax, vals, title, ylabel in zip(
        axes,
        [fid, kid, clip],
        ["FID (lower is better)", "KID (lower is better)", "CLIP Score (higher is better)"],
        ["FID", "KID", "CLIP Score"]
    ):
        bars = ax.bar(models, vals, color=colors, width=0.4)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=10)

    plt.suptitle("A07: Negative Prompt Ablation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/A07_negative_prompt.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved A07")

# ---------------------------------------------------------------------------
# A08 — Architecture Comparison
# ---------------------------------------------------------------------------
def plot_A08():
    models = ["SD v1.5\nBaseline", "SD v1.5\n+ LoRA", "SDXL\nZero-shot"]
    fid    = [87.25, 85.49, 87.25]
    kid    = [0.0357, 0.0310, 0.0356]
    clip   = [31.34, 31.38, 31.34]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for ax, vals, title, ylabel in zip(
        axes,
        [fid, kid, clip],
        ["FID (lower is better)", "KID (lower is better)", "CLIP Score (higher is better)"],
        ["FID", "KID", "CLIP Score"]
    ):
        bars = ax.bar(models, vals, color=colors, width=0.4)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=10)

    plt.suptitle("A08: Architecture Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/A08_architectures.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved A08")


if __name__ == "__main__":
    plot_A01()
    plot_A02()
    plot_A03()
    plot_A04()
    plot_A06()
    plot_A07()
    plot_A08()
    print(f"\nAll plots saved to {OUT_DIR}/")
