"""Generate publication-quality figures from saved experiment files."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (3.5, 2.8),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
})

COLORS = {
    "td3": "#1f77b4",
    "maddpg": "#ff7f0e",
    "noma": "#2ca02c",
    "sdma": "#d62728",
    "no_rs": "#9467bd",
}


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(name + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(name + ".png", dpi=300, bbox_inches="tight")


def figure_sum_rate_vs_snr() -> None:
    data = np.load("sum_rate_vs_snr.npz")
    fig, ax = plt.subplots()
    ax.errorbar(data["snr_dbm"], data["td3_rsma"], yerr=data["td3_rsma_std"], marker="o", linestyle="-", color=COLORS["td3"], label="TD3-RSMA", capsize=3)
    ax.plot(data["snr_dbm"], data["maddpg_rsma"], marker="s", linestyle="--", color=COLORS["maddpg"], label="MADDPG-RSMA")
    ax.plot(data["snr_dbm"], data["noma"], marker="^", linestyle=":", color=COLORS["noma"], label="NOMA")
    ax.plot(data["snr_dbm"], data["sdma"], marker="d", linestyle="-.", color=COLORS["sdma"], label="SDMA")
    ax.plot(data["snr_dbm"], data["no_rs"], marker="x", linestyle=":", color=COLORS["no_rs"], label="No-RS")
    ax.set_xlabel(r"$P_{\max}$ (dBm)")
    ax.set_ylabel("Sum-rate (bps/Hz)")
    ax.legend(fontsize=9)
    _save(fig, "figure1_sum_rate_vs_snr")


def figure_convergence() -> None:
    data = np.load("convergence.npz")
    td3 = data["td3_curves"]
    maddpg = data["maddpg_curves"]
    episodes = np.arange(1, td3.shape[1] + 1)
    fig, ax = plt.subplots()
    td3_mean, td3_std = td3.mean(axis=0), td3.std(axis=0)
    maddpg_mean, maddpg_std = maddpg.mean(axis=0), maddpg.std(axis=0)
    ax.plot(episodes, td3_mean, label="TD3", color=COLORS["td3"])
    ax.fill_between(episodes, td3_mean - td3_std, td3_mean + td3_std, color=COLORS["td3"], alpha=0.2)
    ax.plot(episodes, maddpg_mean, label="MADDPG", color=COLORS["maddpg"])
    ax.fill_between(episodes, maddpg_mean - maddpg_std, maddpg_mean + maddpg_std, color=COLORS["maddpg"], alpha=0.2)
    ax.axhline(float(np.mean(data["noma"])), color=COLORS["noma"], linestyle=":", label="NOMA")
    ax.axhline(float(np.mean(data["sdma"])), color=COLORS["sdma"], linestyle="-.", label="SDMA")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average sum-rate")
    ax.legend(fontsize=9)
    _save(fig, "figure2_convergence")


def figure_csit_error() -> None:
    data = np.load("sum_rate_vs_csit_error.npz")
    fig, ax = plt.subplots()
    ax.plot(data["csit_error"], data["td3_rsma"], marker="o", linestyle="-", color=COLORS["td3"], label="TD3-RSMA")
    ax.plot(data["csit_error"], data["noma"], marker="^", linestyle=":", color=COLORS["noma"], label="NOMA")
    ax.plot(data["csit_error"], data["sdma"], marker="d", linestyle="-.", color=COLORS["sdma"], label="SDMA")
    ax.set_xlabel(r"$\sigma_e$")
    ax.set_ylabel("Sum-rate (bps/Hz)")
    ax.legend(fontsize=9)
    _save(fig, "figure3_sum_rate_vs_csit_error")


def figure_fairness() -> None:
    data = np.load("fairness.npz")
    fig, ax = plt.subplots()
    ax.plot(data["beta"], data["rsma_fairness"], marker="o", linestyle="-", color=COLORS["td3"], label="RSMA")
    ax.plot(data["beta"], data["noma_fairness"], marker="^", linestyle=":", color=COLORS["noma"], label="NOMA")
    ax.plot(data["beta"], data["sdma_fairness"], marker="d", linestyle="-.", color=COLORS["sdma"], label="SDMA")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Jain's fairness index")
    ax.legend(fontsize=9)
    _save(fig, "figure4_fairness")


def figure_power_and_split_behavior() -> None:
    data = np.load("sum_rate_vs_snr.npz")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axes[0].plot(data["snr_dbm"], data["learned_alpha"], marker="o", linestyle="-", color=COLORS["td3"])
    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel(r"$P_{\max}$ (dBm)")
    axes[0].set_ylabel(r"Learned $\alpha$")
    axes[1].plot(data["snr_dbm"], data["learned_common_split"], marker="s", linestyle="--", color=COLORS["maddpg"])
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel(r"$P_{\max}$ (dBm)")
    axes[1].set_ylabel(r"$c_1 / (c_1 + c_2)$")
    _save(fig, "figure5_power_and_split_behavior")


def write_results_table() -> None:
    """Write a compact numerical summary table."""
    snr = np.load("sum_rate_vs_snr.npz")
    csit = np.load("sum_rate_vs_csit_error.npz")
    fair = np.load("fairness.npz")
    lines = [
        "Key numerical results",
        "=====================",
        f"Best TD3 sum-rate over SNR sweep: {np.max(snr['td3_rsma']):.4f}",
        f"Best MADDPG sum-rate over SNR sweep: {np.max(snr['maddpg_rsma']):.4f}",
        f"Average NOMA sum-rate over SNR sweep: {np.mean(snr['noma']):.4f}",
        f"Average SDMA sum-rate over SNR sweep: {np.mean(snr['sdma']):.4f}",
        f"TD3 robustness at max CSIT error: {csit['td3_rsma'][-1]:.4f}",
        f"NOMA robustness at max CSIT error: {csit['noma'][-1]:.4f}",
        f"SDMA robustness at max CSIT error: {csit['sdma'][-1]:.4f}",
        f"Best RSMA fairness: {np.max(fair['rsma_fairness']):.4f}",
    ]
    with open("results_table.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main() -> None:
    """Generate all paper figures."""
    required = [
        "sum_rate_vs_snr.npz",
        "convergence.npz",
        "sum_rate_vs_csit_error.npz",
        "fairness.npz",
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing experiment files: {missing}")
    figure_sum_rate_vs_snr()
    figure_convergence()
    figure_csit_error()
    figure_fairness()
    figure_power_and_split_behavior()
    write_results_table()


if __name__ == "__main__":
    main()
