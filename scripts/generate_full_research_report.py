import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full research markdown report from full-run CSV outputs.")
    parser.add_argument("--strategy-csv", required=True)
    parser.add_argument("--voxel-csv", required=True)
    parser.add_argument("--neighbor-csv", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-doc", required=True)
    args = parser.parse_args()

    strategy_rows = read_csv(Path(args.strategy_csv))
    voxel_rows = read_csv(Path(args.voxel_csv))
    neighbor_rows = read_csv(Path(args.neighbor_csv))

    baseline = None
    for r in strategy_rows:
        if r.get("strategy_group") == "baseline_adaptive":
            baseline = r
            break
    if baseline is None:
        raise ValueError("baseline_adaptive missing in strategy summary.")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append("# 全量实验报告（前端策略）")
    lines.append("")
    lines.append(f"生成时间：`{now}`")
    lines.append("")
    lines.append("## 1. 实验目标")
    lines.append("")
    lines.append("- 复现实验线并在全量测试集上重跑，得到可用于论文量化表与消融图的硬证据。")
    lines.append("- 覆盖策略重跑、voxel_size扫参、neighbor_radius×min_points正交三类核心补测项。")
    lines.append("")
    lines.append("## 2. 全量策略重跑结果")
    lines.append("")
    lines.append("基线（baseline_adaptive）:")
    lines.append(
        f"- Car/Ped/Cyc/Mean 3D mod = {fmt(float(baseline['car_3d_moderate']))} / "
        f"{fmt(float(baseline['pedestrian_3d_moderate']))} / {fmt(float(baseline['cyclist_3d_moderate']))} / "
        f"{fmt(float(baseline['mean_3d_moderate']))}"
    )
    lines.append("")
    lines.append("| strategy_group | profile | ratio | boost | suppress | car | ped | cyc | mean |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in strategy_rows:
        lines.append(
            f"| {r['strategy_group']} | {r['frontend_profile']} | {float(r['adaptive_feature_max_adjust_ratio']):.2f} | "
            f"{float(r['adaptive_feature_boost_strength']):.2f} | {float(r['adaptive_feature_ground_suppress_strength']):.2f} | "
            f"{fmt(float(r['car_3d_moderate']))} ({float(r['car_3d_moderate_delta_vs_baseline']):+.3f}) | "
            f"{fmt(float(r['pedestrian_3d_moderate']))} ({float(r['pedestrian_3d_moderate_delta_vs_baseline']):+.3f}) | "
            f"{fmt(float(r['cyclist_3d_moderate']))} ({float(r['cyclist_3d_moderate_delta_vs_baseline']):+.3f}) | "
            f"{fmt(float(r['mean_3d_moderate']))} ({float(r['mean_3d_moderate_delta_vs_baseline']):+.3f}) |"
        )
    lines.append("")
    lines.append("```mermaid")
    lines.append("xychart-beta")
    lines.append("    title \"Strategy Car 3D mod\"")
    labels = [r["strategy_group"] for r in strategy_rows]
    car_vals = [round(float(r["car_3d_moderate"]), 3) for r in strategy_rows]
    lines.append("    x-axis [" + ", ".join(labels) + "]")
    ymax = max(car_vals) if car_vals else 1.0
    lines.append(f"    y-axis \"Car 3D mod\" 0 --> {max(1, int(ymax + 5))}")
    lines.append("    bar [" + ", ".join(str(v) for v in car_vals) + "]")
    lines.append("```")
    lines.append("")

    lines.append("## 3. voxel_size扫参（推理期）")
    lines.append("")
    lines.append("| exp_id | voxel_xy | voxel_z | map_3d_mod | car | ped | cyc | exports/frame |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in voxel_rows:
        lines.append(
            f"| {r['exp_id']} | {float(r['voxel_size_xy']):.2f} | {float(r['voxel_size_z']):.2f} | "
            f"{fmt(float(r['map_3d_mod']))} | {fmt(float(r['car_3d_mod']))} | {fmt(float(r['ped_3d_mod']))} | "
            f"{fmt(float(r['cyc_3d_mod']))} | {float(r['exports_per_frame']):.3f} |"
        )
    lines.append("")

    lines.append("## 4. neighbor_radius × min_points 正交")
    lines.append("")
    lines.append("| group | ratio | radius | min_points | drop_ratio | map_3d_mod | ped_3d_mod |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in neighbor_rows:
        drop = r.get("drop_ratio", "")
        drop_str = f"{float(drop):.4f}" if drop not in ("", None) else ""
        lines.append(
            f"| {r['grid_group']} | {float(r['adaptive_feature_max_adjust_ratio']):.2f} | "
            f"{int(float(r['neighbor_radius_voxels']))} | {int(float(r['min_neighbor_points']))} | "
            f"{drop_str} | {fmt(float(r['map_3d_mod']))} | {fmt(float(r['ped_3d_mod']))} |"
        )
    lines.append("")

    lines.append("## 5. 产物路径")
    lines.append("")
    lines.append(f"- 策略汇总: `{Path(args.strategy_csv)}`")
    lines.append(f"- voxel扫参: `{Path(args.voxel_csv)}`")
    lines.append(f"- neighbor正交: `{Path(args.neighbor_csv)}`")
    lines.append("")

    out_md = Path(args.out_md)
    out_doc = Path(args.out_doc)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_doc.parent.mkdir(parents=True, exist_ok=True)

    content = "\n".join(lines) + "\n"
    out_md.write_text(content, encoding="utf-8")
    out_doc.write_text(content, encoding="utf-8")

    print(f"[saved] {out_md}")
    print(f"[saved] {out_doc}")


if __name__ == "__main__":
    main()
