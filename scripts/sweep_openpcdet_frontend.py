import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PRESET_CONFIGS: Dict[str, Dict[str, object]] = {
    # Keep CLI fully manual.
    "none": {},
    # Recommended balanced setup:
    # keep Car gains without strongly hurting Ped/Cyc.
    "car_gain_safe_200": {
        "limit": 200,
        "profile_list": "adaptive,adaptive_enhance",
        "adaptive_enable_feature_enhance": False,
        "adaptive_enable_denoise": False,
        "adaptive_feature_grid_size_xy": 0.8,
        "adaptive_feature_min_relative_height": 0.2,
        "adaptive_feature_relative_height_scale": 1.2,
        "adaptive_feature_boost_strength": 0.2,
        "adaptive_feature_ground_suppress_strength": 0.03,
        "adaptive_feature_max_adjust_ratio": 0.35,
        "run_prefix": "sweep_car_gain_safe_200",
    },
    # Aggressive setup:
    # larger Car boost, higher risk for Ped/Cyc degradation.
    "car_gain_aggressive_200": {
        "limit": 200,
        "profile_list": "adaptive,adaptive_enhance",
        "adaptive_enable_feature_enhance": False,
        "adaptive_enable_denoise": False,
        "adaptive_feature_grid_size_xy": 0.8,
        "adaptive_feature_min_relative_height": 0.2,
        "adaptive_feature_relative_height_scale": 1.2,
        "adaptive_feature_boost_strength": 0.2,
        "adaptive_feature_ground_suppress_strength": 0.03,
        "adaptive_feature_max_adjust_ratio": 0.6,
        "run_prefix": "sweep_car_gain_aggressive_200",
    },
    # Full-on setup:
    # maximizes enhancement activation; use for upper-bound ablation only.
    "car_gain_full_200": {
        "limit": 200,
        "profile_list": "adaptive,adaptive_enhance",
        "adaptive_enable_feature_enhance": False,
        "adaptive_enable_denoise": False,
        "adaptive_feature_grid_size_xy": 0.8,
        "adaptive_feature_min_relative_height": 0.2,
        "adaptive_feature_relative_height_scale": 1.2,
        "adaptive_feature_boost_strength": 0.2,
        "adaptive_feature_ground_suppress_strength": 0.03,
        "adaptive_feature_max_adjust_ratio": 1.0,
        "run_prefix": "sweep_car_gain_full_200",
    },
}


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def tag_float(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_eval_summary(eval_json_path: Path) -> Dict[str, float]:
    payload = json.loads(eval_json_path.read_text(encoding="utf-8"))
    ap = payload["ap_summary"]
    car = float(ap["car"]["3d"]["moderate"])
    ped = float(ap["pedestrian"]["3d"]["moderate"])
    cyc = float(ap["cyclist"]["3d"]["moderate"])
    mean_3d = float((car + ped + cyc) / 3.0)
    return {
        "car_3d_moderate": car,
        "pedestrian_3d_moderate": ped,
        "cyclist_3d_moderate": cyc,
        "mean_3d_moderate": mean_3d,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grid sweep for OpenPCDet frontend profiles and parameters: "
            "openpcdet_infer_kitti.py + kitti_devkit_eval.py."
        )
    )
    parser.add_argument("--kitti-root", required=True, help="KITTI root with training/ subfolders.")
    parser.add_argument("--openpcdet-root", required=True, help="OpenPCDet repository root.")
    parser.add_argument("--cfg-file", required=True, help="OpenPCDet model config yaml.")
    parser.add_argument("--ckpt", required=True, help="OpenPCDet checkpoint path.")
    parser.add_argument("--out-root", default="runs/sweep_openpcdet_frontend", help="Output root for sweep runs.")
    parser.add_argument("--run-prefix", default="sweep_openpcdet_frontend", help="Prefix for eval run names.")
    parser.add_argument("--limit", type=int, default=200, help="Frame limit per run (0 = full).")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Inference device.")
    parser.add_argument("--score-thresh", type=float, default=0.1, help="OpenPCDet score threshold.")
    parser.add_argument(
        "--profile-list",
        default="none,safe,adaptive,adaptive_ground,adaptive_enhance,denoise",
        help="Comma-separated frontend profiles to test.",
    )
    parser.add_argument(
        "--denoise-voxel-size-list",
        default="0.2,0.25,0.3",
        help="Comma-separated voxel sizes for denoise profile.",
    )
    parser.add_argument(
        "--denoise-neighbor-radius-list",
        default="1",
        help="Comma-separated neighbor radius (voxels) for denoise profile.",
    )
    parser.add_argument(
        "--denoise-min-neighbor-points-list",
        default="3,4,5",
        help="Comma-separated min neighbor points for denoise profile.",
    )
    parser.add_argument(
        "--temporal-pose-file-list",
        default="",
        help=(
            "Comma-separated pose files for temporal_3frame profile. "
            "Use empty to test no-pose temporal stacking."
        ),
    )
    parser.add_argument(
        "--temporal-allow-missing-pose",
        action="store_true",
        help="Allow temporal stacking without pose compensation.",
    )
    parser.add_argument(
        "--adaptive-enable-denoise",
        action="store_true",
        help="Enable guarded denoise for adaptive profile runs.",
    )
    parser.add_argument(
        "--adaptive-grid-denoise-params",
        action="store_true",
        help=(
            "When profile includes adaptive, expand adaptive experiments by "
            "denoise voxel/radius/min-points lists (orthogonal sweep)."
        ),
    )
    parser.add_argument(
        "--adaptive-enable-feature-enhance",
        action="store_true",
        help="Enable local-height-based feature enhancement in adaptive profile runs.",
    )
    parser.add_argument(
        "--adaptive-max-denoise-drop-ratio",
        type=float,
        default=0.02,
        help="Max drop ratio to apply denoise in adaptive profile.",
    )
    parser.add_argument(
        "--adaptive-feature-grid-size-xy",
        type=float,
        default=0.8,
        help="XY grid size (meters) for local-height feature enhancement.",
    )
    parser.add_argument(
        "--adaptive-feature-min-relative-height",
        type=float,
        default=0.2,
        help="Minimum relative local height (meters) to start foreground boosting.",
    )
    parser.add_argument(
        "--adaptive-feature-relative-height-scale",
        type=float,
        default=1.2,
        help="Relative height scale (meters) for foreground boosting saturation.",
    )
    parser.add_argument(
        "--adaptive-feature-boost-strength",
        type=float,
        default=0.2,
        help="Foreground intensity boost strength for feature enhancement.",
    )
    parser.add_argument(
        "--adaptive-feature-ground-suppress-strength",
        type=float,
        default=0.03,
        help="Ground intensity suppression strength for feature enhancement.",
    )
    parser.add_argument(
        "--adaptive-feature-max-adjust-ratio",
        type=float,
        default=0.35,
        help="Skip feature enhancement when adjusted-point ratio exceeds this threshold.",
    )
    parser.add_argument(
        "--adaptive-intensity-lower-percentile",
        type=float,
        default=1.0,
        help="Lower percentile used by adaptive intensity auto-repair.",
    )
    parser.add_argument(
        "--adaptive-intensity-upper-percentile",
        type=float,
        default=99.0,
        help="Upper percentile used by adaptive intensity auto-repair.",
    )
    parser.add_argument(
        "--adaptive-intensity-trigger-low",
        type=float,
        default=-0.1,
        help="Lower trigger bound for adaptive intensity auto-repair.",
    )
    parser.add_argument(
        "--adaptive-intensity-trigger-high",
        type=float,
        default=1.5,
        help="Upper trigger bound for adaptive intensity auto-repair.",
    )
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for subprocess calls.")
    parser.add_argument("--exe-path", default="cpp/evaluate_object", help="KITTI evaluator executable path.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_CONFIGS.keys()),
        default="none",
        help=(
            "Apply a research preset configuration. "
            "Preset values override the corresponding CLI options."
        ),
    )
    parser.add_argument(
        "--compile-eval",
        action="store_true",
        help="Compile evaluator in the first eval run (recommended on fresh env).",
    )
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> None:
    preset_name = str(args.preset).strip().lower()
    config = PRESET_CONFIGS.get(preset_name)
    if config is None:
        raise ValueError(f"Unknown preset: {preset_name}")
    for key, value in config.items():
        setattr(args, key, value)


def main() -> None:
    args = parse_args()
    apply_preset(args)
    if str(args.preset).lower() != "none":
        print(f"[preset] {args.preset}")

    repo_root = Path(__file__).resolve().parent.parent
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    kitti_root = Path(args.kitti_root).expanduser().resolve()
    openpcdet_root = Path(args.openpcdet_root).expanduser().resolve()
    cfg_file = Path(args.cfg_file).expanduser().resolve()
    ckpt_file = Path(args.ckpt).expanduser().resolve()

    bin_dir = kitti_root / "training" / "velodyne"
    calib_dir = kitti_root / "training" / "calib"
    label_dir = kitti_root / "training" / "label_2"

    missing = [p for p in (bin_dir, calib_dir, label_dir, cfg_file, ckpt_file) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths: " + ", ".join(str(p) for p in missing))

    profiles = parse_str_list(args.profile_list)
    denoise_voxel_sizes = parse_float_list(args.denoise_voxel_size_list)
    denoise_neighbor_radii = parse_int_list(args.denoise_neighbor_radius_list)
    denoise_min_points = parse_int_list(args.denoise_min_neighbor_points_list)

    temporal_pose_files = parse_str_list(args.temporal_pose_file_list)
    if not temporal_pose_files:
        temporal_pose_files = [""]

    experiments: List[Dict[str, object]] = []
    for profile in profiles:
        p = profile.strip().lower()
        if p in {"none", "safe", "adaptive_ground", "adaptive_enhance"}:
            experiments.append(
                {
                    "frontend_profile": p,
                    "denoise_voxel_size": 0.25,
                    "denoise_neighbor_radius": 1,
                    "denoise_min_neighbor_points": 4,
                    "temporal_pose_file": "",
                    "tag_suffix": p,
                }
            )
        elif p == "adaptive":
            if args.adaptive_grid_denoise_params and args.adaptive_enable_denoise:
                for vox, rad, mp in itertools.product(
                    denoise_voxel_sizes, denoise_neighbor_radii, denoise_min_points
                ):
                    experiments.append(
                        {
                            "frontend_profile": "adaptive",
                            "denoise_voxel_size": float(vox),
                            "denoise_neighbor_radius": int(rad),
                            "denoise_min_neighbor_points": int(mp),
                            "temporal_pose_file": "",
                            "tag_suffix": (
                                "adaptive"
                                f"_v{tag_float(float(vox))}"
                                f"_r{int(rad)}_m{int(mp)}"
                            ),
                        }
                    )
            else:
                experiments.append(
                    {
                        "frontend_profile": "adaptive",
                        "denoise_voxel_size": 0.25,
                        "denoise_neighbor_radius": 1,
                        "denoise_min_neighbor_points": 4,
                        "temporal_pose_file": "",
                        "tag_suffix": "adaptive",
                    }
                )
        elif p == "denoise":
            for vox, rad, mp in itertools.product(
                denoise_voxel_sizes, denoise_neighbor_radii, denoise_min_points
            ):
                experiments.append(
                    {
                        "frontend_profile": "denoise",
                        "denoise_voxel_size": float(vox),
                        "denoise_neighbor_radius": int(rad),
                        "denoise_min_neighbor_points": int(mp),
                        "temporal_pose_file": "",
                        "tag_suffix": (
                            f"denoise_v{tag_float(float(vox))}"
                            f"_r{int(rad)}_m{int(mp)}"
                        ),
                    }
                )
        elif p == "temporal_3frame":
            for pose in temporal_pose_files:
                pose_path = ""
                pose_tag = "nopose"
                if pose:
                    pose_path = str(Path(pose).expanduser().resolve())
                    if not Path(pose_path).exists():
                        raise FileNotFoundError(f"Temporal pose file not found: {pose_path}")
                    pose_tag = Path(pose_path).stem
                experiments.append(
                    {
                        "frontend_profile": "temporal_3frame",
                        "denoise_voxel_size": 0.25,
                        "denoise_neighbor_radius": 1,
                        "denoise_min_neighbor_points": 4,
                        "temporal_pose_file": pose_path,
                        "tag_suffix": f"temporal_{pose_tag}",
                    }
                )
        else:
            raise ValueError(f"Unsupported profile in --profile-list: {profile}")

    if not experiments:
        raise ValueError("No experiments generated. Check --profile-list.")

    results = []
    openpcdet_tools = openpcdet_root / "tools"
    openpcdet_cwd = openpcdet_tools if openpcdet_tools.exists() else openpcdet_root
    compiled = False

    for idx, exp in enumerate(experiments):
        run_tag = f"{idx:03d}_{exp['tag_suffix']}"
        run_dir = out_root / run_tag
        pred_dir = run_dir / "predictions" / "data"
        pred_dir.mkdir(parents=True, exist_ok=True)

        export_cmd = [
            args.python_bin,
            str(repo_root / "openpcdet_infer_kitti.py"),
            "--openpcdet-root",
            str(openpcdet_root),
            "--cfg-file",
            str(cfg_file),
            "--ckpt",
            str(ckpt_file),
            "--bin-dir",
            str(bin_dir),
            "--calib-dir",
            str(calib_dir),
            "--out-dir",
            str(pred_dir),
            "--score-thresh",
            str(args.score_thresh),
            "--device",
            args.device,
            "--frontend-profile",
            str(exp["frontend_profile"]),
            "--frontend-denoise-voxel-size",
            str(exp["denoise_voxel_size"]),
            "--frontend-denoise-neighbor-radius-voxels",
            str(exp["denoise_neighbor_radius"]),
            "--frontend-denoise-min-neighbor-points",
            str(exp["denoise_min_neighbor_points"]),
            "--frontend-adaptive-max-denoise-drop-ratio",
            str(args.adaptive_max_denoise_drop_ratio),
            "--frontend-adaptive-intensity-lower-percentile",
            str(args.adaptive_intensity_lower_percentile),
            "--frontend-adaptive-intensity-upper-percentile",
            str(args.adaptive_intensity_upper_percentile),
            "--frontend-adaptive-intensity-trigger-low",
            str(args.adaptive_intensity_trigger_low),
            "--frontend-adaptive-intensity-trigger-high",
            str(args.adaptive_intensity_trigger_high),
            "--frontend-adaptive-feature-grid-size-xy",
            str(args.adaptive_feature_grid_size_xy),
            "--frontend-adaptive-feature-min-relative-height",
            str(args.adaptive_feature_min_relative_height),
            "--frontend-adaptive-feature-relative-height-scale",
            str(args.adaptive_feature_relative_height_scale),
            "--frontend-adaptive-feature-boost-strength",
            str(args.adaptive_feature_boost_strength),
            "--frontend-adaptive-feature-ground-suppress-strength",
            str(args.adaptive_feature_ground_suppress_strength),
            "--frontend-adaptive-feature-max-adjust-ratio",
            str(args.adaptive_feature_max_adjust_ratio),
        ]
        if args.adaptive_enable_denoise:
            export_cmd.append("--frontend-adaptive-enable-denoise")
        if args.adaptive_enable_feature_enhance:
            export_cmd.append("--frontend-adaptive-enable-feature-enhance")
        if args.temporal_allow_missing_pose:
            export_cmd.append("--frontend-temporal-allow-missing-pose")
        if args.limit > 0:
            export_cmd.extend(["--limit", str(args.limit)])
        temporal_pose_file = str(exp["temporal_pose_file"])
        if temporal_pose_file:
            export_cmd.extend(["--frontend-temporal-pose-file", temporal_pose_file])
        run_cmd(export_cmd, openpcdet_cwd)

        eval_run_name = f"{args.run_prefix}_{run_tag}"
        eval_cmd = [
            args.python_bin,
            "kitti_devkit_eval.py",
            "--label-dir",
            str(label_dir),
            "--pred-dir",
            str(pred_dir),
            "--run-name",
            eval_run_name,
            "--exe-path",
            args.exe_path,
        ]
        if args.limit > 0:
            eval_cmd.extend(["--n-test-images", str(args.limit)])
        if args.compile_eval and not compiled:
            eval_cmd.append("--compile")
            compiled = True
        run_cmd(eval_cmd, repo_root)

        export_summary_path = pred_dir.parent / "openpcdet_export_summary.json"
        eval_summary_path = repo_root / "results" / eval_run_name / "kitti_eval_summary.json"

        export_summary = json.loads(export_summary_path.read_text(encoding="utf-8"))
        eval_summary = load_eval_summary(eval_summary_path)

        row = {
            "preset": str(args.preset),
            "run_tag": run_tag,
            "frontend_profile": exp["frontend_profile"],
            "denoise_voxel_size": exp["denoise_voxel_size"],
            "denoise_neighbor_radius": exp["denoise_neighbor_radius"],
            "denoise_min_neighbor_points": exp["denoise_min_neighbor_points"],
            "temporal_pose_file": temporal_pose_file,
            "temporal_allow_missing_pose": bool(args.temporal_allow_missing_pose),
            "adaptive_enable_denoise": bool(args.adaptive_enable_denoise),
            "adaptive_enable_feature_enhance": bool(
                args.adaptive_enable_feature_enhance
                or str(exp["frontend_profile"]) == "adaptive_enhance"
            ),
            "adaptive_max_denoise_drop_ratio": float(args.adaptive_max_denoise_drop_ratio),
            "adaptive_feature_grid_size_xy": float(args.adaptive_feature_grid_size_xy),
            "adaptive_feature_min_relative_height": float(
                args.adaptive_feature_min_relative_height
            ),
            "adaptive_feature_relative_height_scale": float(
                args.adaptive_feature_relative_height_scale
            ),
            "adaptive_feature_boost_strength": float(args.adaptive_feature_boost_strength),
            "adaptive_feature_ground_suppress_strength": float(
                args.adaptive_feature_ground_suppress_strength
            ),
            "adaptive_feature_max_adjust_ratio": float(args.adaptive_feature_max_adjust_ratio),
            "adaptive_intensity_lower_percentile": float(args.adaptive_intensity_lower_percentile),
            "adaptive_intensity_upper_percentile": float(args.adaptive_intensity_upper_percentile),
            "adaptive_intensity_trigger_low": float(args.adaptive_intensity_trigger_low),
            "adaptive_intensity_trigger_high": float(args.adaptive_intensity_trigger_high),
            "num_frames": int(export_summary["num_frames"]),
            "total_exported": int(export_summary["total_exported"]),
            "car_3d_moderate": eval_summary["car_3d_moderate"],
            "pedestrian_3d_moderate": eval_summary["pedestrian_3d_moderate"],
            "cyclist_3d_moderate": eval_summary["cyclist_3d_moderate"],
            "mean_3d_moderate": eval_summary["mean_3d_moderate"],
            "export_summary_json": str(export_summary_path),
            "eval_summary_json": str(eval_summary_path),
        }
        results.append(row)

    results.sort(key=lambda x: float(x["mean_3d_moderate"]), reverse=True)
    summary_json = out_root / "sweep_summary.json"
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary_csv = out_root / "sweep_summary.csv"
    fieldnames = [
        "preset",
        "run_tag",
        "frontend_profile",
        "denoise_voxel_size",
        "denoise_neighbor_radius",
        "denoise_min_neighbor_points",
        "temporal_pose_file",
        "temporal_allow_missing_pose",
        "adaptive_enable_denoise",
        "adaptive_enable_feature_enhance",
        "adaptive_max_denoise_drop_ratio",
        "adaptive_feature_grid_size_xy",
        "adaptive_feature_min_relative_height",
        "adaptive_feature_relative_height_scale",
        "adaptive_feature_boost_strength",
        "adaptive_feature_ground_suppress_strength",
        "adaptive_feature_max_adjust_ratio",
        "adaptive_intensity_lower_percentile",
        "adaptive_intensity_upper_percentile",
        "adaptive_intensity_trigger_low",
        "adaptive_intensity_trigger_high",
        "num_frames",
        "total_exported",
        "car_3d_moderate",
        "pedestrian_3d_moderate",
        "cyclist_3d_moderate",
        "mean_3d_moderate",
        "export_summary_json",
        "eval_summary_json",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"[saved] {summary_json}")
    print(f"[saved] {summary_csv}")
    if results:
        print("[best]", json.dumps(results[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
