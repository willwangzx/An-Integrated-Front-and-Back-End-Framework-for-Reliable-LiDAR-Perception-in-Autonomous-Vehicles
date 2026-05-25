#!/usr/bin/env python3
"""Build paper-ready figures and tables from existing experiment artifacts.

Usage:
  .venv/bin/python scripts/build_paper_figures_tables.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = REPO_ROOT / "runs"
FULL_SUITE_DIR = RUNS_DIR / "full_research_suite_v1"

FIG_DIR = REPO_ROOT / "docs" / "figures" / "paper_research"
TAB_DIR = REPO_ROOT / "docs" / "tables" / "paper_research"
DOC_PATH = REPO_ROOT / "docs" / "PAPER_FIGURES_TABLES_ZH.md"


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def to_markdown_table(rows: List[Dict], fieldnames: List[str]) -> str:
    header = "| " + " | ".join(fieldnames) + " |"
    sep = "|" + "|".join(["---"] * len(fieldnames)) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in fieldnames) + " |")
    return "\n".join(lines)


def ap_mod(eval_json: Dict, cls: str, metric: str) -> float:
    return float(eval_json["ap_summary"][cls][metric]["moderate"])


def fmt(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}"


def build_table1_main_results() -> Path:
    runs = [
        ("OpenPCDet(PointPillar)", RESULTS_DIR / "openpcdet_pointpillar_7481" / "kitti_eval_summary.json"),
        ("Heuristic Baseline", RESULTS_DIR / "full_eval_7481" / "kitti_eval_summary.json"),
        ("Heuristic Improved", RESULTS_DIR / "full_eval_improved_7481" / "kitti_eval_summary.json"),
        ("Heuristic Recall-Reliable", RESULTS_DIR / "full_eval_recall_reliable_7481" / "kitti_eval_summary.json"),
    ]

    rows: List[Dict] = []
    for name, p in runs:
        e = read_json(p)
        car3d = ap_mod(e, "car", "3d")
        ped3d = ap_mod(e, "pedestrian", "3d")
        cyc3d = ap_mod(e, "cyclist", "3d")
        map3d = (car3d + ped3d + cyc3d) / 3.0

        car_bev = ap_mod(e, "car", "bev")
        ped_bev = ap_mod(e, "pedestrian", "bev")
        cyc_bev = ap_mod(e, "cyclist", "bev")
        map_bev = (car_bev + ped_bev + cyc_bev) / 3.0

        car_2d = ap_mod(e, "car", "2d")
        ped_2d = ap_mod(e, "pedestrian", "2d")
        cyc_2d = ap_mod(e, "cyclist", "2d")
        map_2d = (car_2d + ped_2d + cyc_2d) / 3.0

        rows.append(
            {
                "method": name,
                "n_test_images": int(e.get("n_test_images", 0)),
                "car_3d_mod": fmt(car3d),
                "ped_3d_mod": fmt(ped3d),
                "cyc_3d_mod": fmt(cyc3d),
                "map_3d_mod": fmt(map3d),
                "map_bev_mod": fmt(map_bev),
                "map_2d_mod": fmt(map_2d),
            }
        )

    fields = [
        "method",
        "n_test_images",
        "car_3d_mod",
        "ped_3d_mod",
        "cyc_3d_mod",
        "map_3d_mod",
        "map_bev_mod",
        "map_2d_mod",
    ]
    csv_path = TAB_DIR / "table1_main_results_7481.csv"
    md_path = TAB_DIR / "table1_main_results_7481.md"
    write_csv(csv_path, rows, fields)
    md_path.write_text(to_markdown_table(rows, fields) + "\n", encoding="utf-8")
    return md_path


def build_1000_rows() -> List[Dict]:
    runs = [
        (
            "baseline",
            "0.35",
            RESULTS_DIR / "aggr_1000_baseline_adaptive_000_adaptive" / "kitti_eval_summary.json",
            RUNS_DIR / "aggressive_matrix_1000_v1" / "baseline_adaptive" / "000_adaptive" / "predictions" / "openpcdet_export_summary.json",
        ),
        (
            "r=0.6",
            "0.60",
            RESULTS_DIR / "aggr_1000_aggressive_r0p6_000_adaptive_enhance" / "kitti_eval_summary.json",
            RUNS_DIR / "aggressive_matrix_1000_v1" / "aggressive_r0p6" / "000_adaptive_enhance" / "predictions" / "openpcdet_export_summary.json",
        ),
        (
            "r=0.8",
            "0.80",
            RESULTS_DIR / "aggr_1000_aggressive_r0p8_000_adaptive_enhance" / "kitti_eval_summary.json",
            RUNS_DIR / "aggressive_matrix_1000_v1" / "aggressive_r0p8" / "000_adaptive_enhance" / "predictions" / "openpcdet_export_summary.json",
        ),
    ]

    rows: List[Dict] = []
    for name, ratio, eval_path, export_path in runs:
        e = read_json(eval_path)
        x = read_json(export_path)
        car = ap_mod(e, "car", "3d")
        ped = ap_mod(e, "pedestrian", "3d")
        cyc = ap_mod(e, "cyclist", "3d")
        mean = (car + ped + cyc) / 3.0
        rows.append(
            {
                "strategy": name,
                "ratio": ratio,
                "n_frames": int(e.get("n_test_images", 0)),
                "car_3d_mod": car,
                "ped_3d_mod": ped,
                "cyc_3d_mod": cyc,
                "mean_3d_mod": mean,
                "total_exported": int(x.get("total_exported", 0)),
            }
        )
    return rows


def build_table2_1000() -> Path:
    rows_raw = build_1000_rows()
    rows = []
    for r in rows_raw:
        rows.append(
            {
                "strategy": r["strategy"],
                "ratio": r["ratio"],
                "n_frames": r["n_frames"],
                "car_3d_mod": fmt(float(r["car_3d_mod"])),
                "ped_3d_mod": fmt(float(r["ped_3d_mod"])),
                "cyc_3d_mod": fmt(float(r["cyc_3d_mod"])),
                "mean_3d_mod": fmt(float(r["mean_3d_mod"])),
                "total_exported": int(r["total_exported"]),
            }
        )

    fields = [
        "strategy",
        "ratio",
        "n_frames",
        "car_3d_mod",
        "ped_3d_mod",
        "cyc_3d_mod",
        "mean_3d_mod",
        "total_exported",
    ]
    csv_path = TAB_DIR / "table2_1000_aggressive_compare.csv"
    md_path = TAB_DIR / "table2_1000_aggressive_compare.md"
    write_csv(csv_path, rows, fields)
    md_path.write_text(to_markdown_table(rows, fields) + "\n", encoding="utf-8")
    return md_path


def build_table3_strategy() -> Path:
    src = FULL_SUITE_DIR / "strategy_rerun" / "full_strategy_summary.csv"
    rows_src = read_csv(src)
    rows = []
    for r in rows_src:
        rows.append(
            {
                "strategy_group": r["strategy_group"],
                "profile": r["frontend_profile"],
                "ratio": fmt(float(r["adaptive_feature_max_adjust_ratio"]), 2),
                "car_3d_mod": fmt(float(r["car_3d_moderate"])),
                "ped_3d_mod": fmt(float(r["pedestrian_3d_moderate"])),
                "cyc_3d_mod": fmt(float(r["cyclist_3d_moderate"])),
                "mean_3d_mod": fmt(float(r["mean_3d_moderate"])),
            }
        )
    fields = [
        "strategy_group",
        "profile",
        "ratio",
        "car_3d_mod",
        "ped_3d_mod",
        "cyc_3d_mod",
        "mean_3d_mod",
    ]
    csv_path = TAB_DIR / "table3_full_strategy_rerun.csv"
    md_path = TAB_DIR / "table3_full_strategy_rerun.md"
    write_csv(csv_path, rows, fields)
    md_path.write_text(to_markdown_table(rows, fields) + "\n", encoding="utf-8")
    return md_path


def build_table4_voxel_sweep() -> Path:
    src = FULL_SUITE_DIR / "voxel_sweep" / "voxel_sweep_table.csv"
    rows_src = read_csv(src)
    rows = []
    for r in rows_src:
        rows.append(
            {
                "exp_id": r["exp_id"],
                "voxel_xy": fmt(float(r["voxel_size_xy"]), 2),
                "map_3d_mod": fmt(float(r["map_3d_mod"])),
                "car_3d_mod": fmt(float(r["car_3d_mod"])),
                "ped_3d_mod": fmt(float(r["ped_3d_mod"])),
                "cyc_3d_mod": fmt(float(r["cyc_3d_mod"])),
                "exports_per_frame": fmt(float(r["exports_per_frame"])),
            }
        )
    fields = [
        "exp_id",
        "voxel_xy",
        "map_3d_mod",
        "car_3d_mod",
        "ped_3d_mod",
        "cyc_3d_mod",
        "exports_per_frame",
    ]
    csv_path = TAB_DIR / "table4_voxel_size_sweep.csv"
    md_path = TAB_DIR / "table4_voxel_size_sweep.md"
    write_csv(csv_path, rows, fields)
    md_path.write_text(to_markdown_table(rows, fields) + "\n", encoding="utf-8")
    return md_path


def build_table5_proxy() -> Path:
    src = FULL_SUITE_DIR / "voxel_sweep" / "voxel_sweep_table.csv"
    vox_rows = read_csv(src)

    # Approximate KITTI box priors: (length, width, height) in meters.
    priors = {
        "Car": (3.9, 1.6, 1.56),
        "Pedestrian": (0.8, 0.6, 1.73),
        "Cyclist": (1.76, 0.6, 1.73),
    }

    rows = []
    for vr in vox_rows:
        vxy = float(vr["voxel_size_xy"])
        vz = float(vr["voxel_size_z"])
        q2d = math.sqrt(2.0) * vxy / 2.0
        for cls, (l, w, h) in priors.items():
            occ2d = math.ceil(l / vxy) * math.ceil(w / vxy)
            occ3d = occ2d * math.ceil(h / vz)
            rows.append(
                {
                    "voxel_xy": fmt(vxy, 2),
                    "voxel_z": fmt(vz, 2),
                    "class": cls,
                    "occupied_cells_2d": occ2d,
                    "occupied_cells_3d": occ3d,
                    "quantization_upper_bound_2d_m": fmt(q2d, 4),
                }
            )

    fields = [
        "voxel_xy",
        "voxel_z",
        "class",
        "occupied_cells_2d",
        "occupied_cells_3d",
        "quantization_upper_bound_2d_m",
    ]
    csv_path = TAB_DIR / "table5_proxy_quantization_analysis.csv"
    md_path = TAB_DIR / "table5_proxy_quantization_analysis.md"
    write_csv(csv_path, rows, fields)
    md_path.write_text(to_markdown_table(rows, fields) + "\n", encoding="utf-8")
    return md_path


def figure1_system_framework(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.02, 0.56, 0.13, 0.22, "Raw Point Cloud"),
        (0.18, 0.56, 0.16, 0.22, "Frontend Preprocessing\n/ Voxel-related Operator"),
        (0.37, 0.56, 0.14, 0.22, "PointPillars\nEncoder"),
        (0.54, 0.56, 0.14, 0.22, "BEV\nScattering"),
        (0.71, 0.56, 0.12, 0.22, "Detection\nHead"),
        (0.85, 0.56, 0.11, 0.22, "KITTI\nExport"),
        (0.85, 0.18, 0.11, 0.22, "Devkit\nEvaluation"),
    ]

    for x, y, w, h, t in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#f2f6ff", edgecolor="#2b4f81", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=10)

    branch = plt.Rectangle((0.56, 0.18), 0.24, 0.22, facecolor="#fff4e6", edgecolor="#aa5a00", linewidth=2)
    ax.add_patch(branch)
    ax.text(
        0.68,
        0.29,
        "Heuristic Backend\n(Reliability Branch)\nNot Main Detector",
        ha="center",
        va="center",
        fontsize=10,
        color="#7a3d00",
    )

    arrows = [
        ((0.15, 0.67), (0.18, 0.67)),
        ((0.34, 0.67), (0.37, 0.67)),
        ((0.51, 0.67), (0.54, 0.67)),
        ((0.68, 0.67), (0.71, 0.67)),
        ((0.83, 0.67), (0.85, 0.67)),
        ((0.905, 0.56), (0.905, 0.40)),
        ((0.905, 0.40), (0.905, 0.40)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2, color="#224b7a"))

    ax.annotate(
        "",
        xy=(0.62, 0.40),
        xytext=(0.77, 0.56),
        arrowprops=dict(arrowstyle="->", lw=2, linestyle="--", color="#aa5a00"),
    )
    ax.annotate(
        "",
        xy=(0.85, 0.58),
        xytext=(0.80, 0.29),
        arrowprops=dict(arrowstyle="->", lw=2, linestyle="--", color="#aa5a00"),
    )

    ax.text(0.03, 0.90, "Main Detection Path", fontsize=10, color="#224b7a")
    ax.plot([0.02, 0.12], [0.885, 0.885], color="#224b7a", lw=2)
    ax.text(0.32, 0.90, "Reliability Branch", fontsize=10, color="#aa5a00")
    ax.plot([0.30, 0.40], [0.885, 0.885], color="#aa5a00", lw=2, linestyle="--")

    ax.set_title("Figure 1. Integrated Framework Overview", fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figure2_quantization(path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    np.random.seed(42)
    pts = np.random.rand(420, 2)
    ax1.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.35, color="#2f6fa3")
    step = 0.1
    for g in np.arange(0, 1 + 1e-9, step):
        ax1.plot([g, g], [0, 1], color="#dddddd", lw=0.8)
        ax1.plot([0, 1], [g, g], color="#dddddd", lw=0.8)

    car = plt.Rectangle((0.18, 0.40), 0.28, 0.18, fill=False, edgecolor="#d04a3a", linewidth=2)
    ped = plt.Rectangle((0.68, 0.52), 0.06, 0.06, fill=False, edgecolor="#2e8b57", linewidth=2)
    ax1.add_patch(car)
    ax1.add_patch(ped)
    ax1.text(0.32, 0.60, "Car", color="#d04a3a", ha="center")
    ax1.text(0.71, 0.61, "Ped", color="#2e8b57", ha="center")
    ax1.set_title("Continuous Point Cloud -> Pillar Grid")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xticks([])
    ax1.set_yticks([])

    voxel = 0.25
    car_cells = math.ceil(3.9 / voxel) * math.ceil(1.6 / voxel)
    ped_cells = math.ceil(0.8 / voxel) * math.ceil(0.6 / voxel)
    cyc_cells = math.ceil(1.76 / voxel) * math.ceil(0.6 / voxel)
    labels = ["Car", "Pedestrian", "Cyclist"]
    vals = [car_cells, ped_cells, cyc_cells]
    colors = ["#d04a3a", "#2e8b57", "#7d5ab5"]
    ax2.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax2.text(i, v + max(vals) * 0.03, str(v), ha="center", fontsize=10)
    ax2.set_title(f"Occupied 2D Cells at Same voxel_xy={voxel:.2f}m")
    ax2.set_ylabel("Occupied Cells (2D)")
    q_upper = math.sqrt(2.0) * voxel / 2.0
    ax2.text(
        0.02,
        0.96,
        f"2D quantization upper bound = {q_upper:.3f} m",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="#f8f8f8", ec="#bbbbbb"),
    )

    fig.suptitle("Figure 2. Quantization Error and Discrete Resolution", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figure3_1000_tradeoff(path: Path) -> None:
    rows = build_1000_rows()
    labels = [r["strategy"] for r in rows]
    car = np.array([r["car_3d_mod"] for r in rows], dtype=float)
    ped = np.array([r["ped_3d_mod"] for r in rows], dtype=float)
    cyc = np.array([r["cyc_3d_mod"] for r in rows], dtype=float)
    mean = np.array([r["mean_3d_mod"] for r in rows], dtype=float)

    x = np.arange(len(labels))
    w = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * w, car, w, label="Car")
    ax.bar(x - 0.5 * w, ped, w, label="Pedestrian")
    ax.bar(x + 0.5 * w, cyc, w, label="Cyclist")
    ax.bar(x + 1.5 * w, mean, w, label="Mean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AP 3D Moderate")
    ax.set_title("Figure 3. 1000-Frame Trade-off under Enhancement Strength")
    ax.legend(ncol=4)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figure4_full_strategy(path: Path) -> None:
    rows = read_csv(FULL_SUITE_DIR / "strategy_rerun" / "full_strategy_summary.csv")
    labels = [r["strategy_group"] for r in rows]
    car = np.array([float(r["car_3d_moderate"]) for r in rows])
    ped = np.array([float(r["pedestrian_3d_moderate"]) for r in rows])
    cyc = np.array([float(r["cyclist_3d_moderate"]) for r in rows])
    mean = np.array([float(r["mean_3d_moderate"]) for r in rows])

    x = np.arange(len(labels))
    w = 0.2

    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.bar(x - 1.5 * w, car, w, label="Car")
    ax.bar(x - 0.5 * w, ped, w, label="Pedestrian")
    ax.bar(x + 0.5 * w, cyc, w, label="Cyclist")
    ax.bar(x + 1.5 * w, mean, w, label="Mean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("AP 3D Moderate")
    ax.set_title("Figure 4. Full 7481-Frame Strategy Rerun Comparison")
    ax.legend(ncol=4)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figure5_voxel_sweep(path: Path) -> None:
    rows = read_csv(FULL_SUITE_DIR / "voxel_sweep" / "voxel_sweep_table.csv")
    rows.sort(key=lambda r: float(r["voxel_size_xy"]))
    x = np.array([float(r["voxel_size_xy"]) for r in rows])
    map3d = np.array([float(r["map_3d_mod"]) for r in rows])
    car = np.array([float(r["car_3d_mod"]) for r in rows])
    ped = np.array([float(r["ped_3d_mod"]) for r in rows])
    cyc = np.array([float(r["cyc_3d_mod"]) for r in rows])

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(x, map3d, marker="o", label="mAP 3D mod")
    ax.plot(x, car, marker="o", label="Car")
    ax.plot(x, ped, marker="o", label="Pedestrian")
    ax.plot(x, cyc, marker="o", label="Cyclist")
    ax.set_xlabel("voxel_xy (m)")
    ax.set_ylabel("AP / mAP 3D Moderate")
    ax.set_title("Figure 5. Denoise Voxel-Size Sweep (7481 frames)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figure6_exports_vs_voxel(path: Path) -> None:
    rows = read_csv(FULL_SUITE_DIR / "voxel_sweep" / "voxel_sweep_table.csv")
    rows.sort(key=lambda r: float(r["voxel_size_xy"]))
    x = np.array([float(r["voxel_size_xy"]) for r in rows])
    y = np.array([float(r["exports_per_frame"]) for r in rows])

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(x, y, marker="o", linewidth=2, color="#385a9c")
    ax.set_xlabel("voxel_xy (m)")
    ax.set_ylabel("exports / frame")
    ax.set_title("Figure 6. Candidate Export Count vs Voxel Size")
    ax.grid(True, linestyle="--", alpha=0.35)
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.08, f"{yi:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figure7_neighbor_heatmap(path: Path) -> None:
    rows = read_csv(FULL_SUITE_DIR / "neighbor_grid" / "neighbor_grid_table.csv")

    radius_vals = sorted({int(float(r["neighbor_radius_voxels"])) for r in rows})
    min_vals = sorted({int(float(r["min_neighbor_points"])) for r in rows})

    map_mat = np.full((len(radius_vals), len(min_vals)), np.nan)
    ped_mat = np.full((len(radius_vals), len(min_vals)), np.nan)

    ridx = {v: i for i, v in enumerate(radius_vals)}
    midx = {v: i for i, v in enumerate(min_vals)}

    for r in rows:
        i = ridx[int(float(r["neighbor_radius_voxels"]))]
        j = midx[int(float(r["min_neighbor_points"]))]
        map_mat[i, j] = float(r["map_3d_mod"])
        ped_mat[i, j] = float(r["ped_3d_mod"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, mat, title in [
        (axes[0], map_mat, "map_3d_mod"),
        (axes[1], ped_mat, "ped_3d_mod"),
    ]:
        im = ax.imshow(mat, cmap="YlOrRd")
        ax.set_xticks(np.arange(len(min_vals)))
        ax.set_yticks(np.arange(len(radius_vals)))
        ax.set_xticklabels(min_vals)
        ax.set_yticklabels(radius_vals)
        ax.set_xlabel("min_points")
        ax.set_ylabel("radius")
        ax.set_title(title)
        for i in range(len(radius_vals)):
            for j in range(len(min_vals)):
                v = mat[i, j]
                txt = "NA" if np.isnan(v) else f"{v:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Figure 7. Neighbor Radius x Min Points Heatmap")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_all_figures() -> List[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    p1 = FIG_DIR / "fig1_system_framework.png"
    figure1_system_framework(p1)
    out.append(p1)

    p2 = FIG_DIR / "fig2_quantization_resolution.png"
    figure2_quantization(p2)
    out.append(p2)

    p3 = FIG_DIR / "fig3_1000_aggressive_tradeoff.png"
    figure3_1000_tradeoff(p3)
    out.append(p3)

    p4 = FIG_DIR / "fig4_full_strategy_7481.png"
    figure4_full_strategy(p4)
    out.append(p4)

    p5 = FIG_DIR / "fig5_voxel_sweep_map.png"
    figure5_voxel_sweep(p5)
    out.append(p5)

    p6 = FIG_DIR / "fig6_exports_vs_voxel.png"
    figure6_exports_vs_voxel(p6)
    out.append(p6)

    p7 = FIG_DIR / "fig7_neighbor_grid_heatmap.png"
    figure7_neighbor_heatmap(p7)
    out.append(p7)

    return out


def write_index_doc(fig_paths: List[Path], table_paths: List[Path]) -> None:
    rel = lambda p: p.relative_to(REPO_ROOT).as_posix()

    lines = [
        "# 论文图表包（基于现有数据）",
        "",
        "本文件汇总图1-图7与表1-表5，对应你给出的正式稿图表需求。",
        "",
        "## 图",
        "",
        "1. 图1 系统框架图（主链 + reliability branch）",
        f"![fig1]({rel(fig_paths[0])})",
        "",
        "2. 图2 量化误差与离散分辨率示意图",
        f"![fig2]({rel(fig_paths[1])})",
        "",
        "3. 图3 1000帧增强强度对三类AP与Mean影响",
        f"![fig3]({rel(fig_paths[2])})",
        "",
        "4. 图4 7481帧全量策略重跑多指标对比",
        f"![fig4]({rel(fig_paths[3])})",
        "",
        "5. 图5 denoise voxel-size sweep 曲线图",
        f"![fig5]({rel(fig_paths[4])})",
        "",
        "6. 图6 exports/frame 与 voxel size 关系图",
        f"![fig6]({rel(fig_paths[5])})",
        "",
        "7. 图7 neighbor_radius × min_points 热力图",
        f"![fig7]({rel(fig_paths[6])})",
        "",
        "## 表",
        "",
        f"1. 表1 主结果表（7481帧，OpenPCDet vs heuristic）：`{rel(table_paths[0])}`",
        f"2. 表2 1000帧增强强度对比表：`{rel(table_paths[1])}`",
        f"3. 表3 全量策略重跑结果表：`{rel(table_paths[2])}`",
        f"4. 表4 voxel-size sweep 表：`{rel(table_paths[3])}`",
        f"5. 表5 proxy analysis 表：`{rel(table_paths[4])}`",
        "",
        "## 说明",
        "",
        "- 图7当前基于已有neighbor结果，仅包含safe_r0p35已完成组合；缺失网格点以NA展示。",
        "- 表5为方法解释用proxy，不是实测性能结果。",
    ]

    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    table_paths = [
        build_table1_main_results(),
        build_table2_1000(),
        build_table3_strategy(),
        build_table4_voxel_sweep(),
        build_table5_proxy(),
    ]
    fig_paths = build_all_figures()
    write_index_doc(fig_paths, table_paths)

    print("[done] Generated paper figures and tables")
    for p in fig_paths:
        print("[figure]", p)
    for p in table_paths:
        print("[table]", p)
    print("[doc]", DOC_PATH)


if __name__ == "__main__":
    main()
