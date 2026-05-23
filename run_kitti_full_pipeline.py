import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenPCDet-only KITTI pipeline in one command: "
            "OpenPCDet export + KITTI devkit eval."
        )
    )
    parser.add_argument(
        "--kitti-root",
        required=True,
        help="KITTI Object root path (must include training/velodyne, calib, label_2).",
    )
    parser.add_argument(
        "--workspace-dir",
        default="runs/kitti_full_pipeline",
        help="Output workspace for predictions and summary.",
    )
    parser.add_argument(
        "--run-name",
        default="openpcdet_full_pipeline",
        help="Run name passed to KITTI devkit evaluator.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of frames for quick validation (0 = all).",
    )
    parser.add_argument(
        "--bash-path",
        default=r"C:\msys64\usr\bin\bash.exe" if platform.system().lower().startswith("win") else "/bin/bash",
        help="Bash path for Windows/MSYS2 compile-run in kitti_devkit_eval.py.",
    )
    parser.add_argument(
        "--openpcdet-root",
        required=True,
        help="OpenPCDet repo path.",
    )
    parser.add_argument(
        "--openpcdet-cfg-file",
        required=True,
        help="OpenPCDet model config.",
    )
    parser.add_argument(
        "--openpcdet-ckpt",
        required=True,
        help="OpenPCDet checkpoint path.",
    )
    parser.add_argument(
        "--openpcdet-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Inference device for OpenPCDet export.",
    )
    parser.add_argument(
        "--openpcdet-frontend-profile",
        choices=[
            "none",
            "safe",
            "denoise",
            "adaptive",
            "adaptive_enhance",
            "adaptive_ground",
            "temporal_3frame",
        ],
        default="adaptive",
        help=(
            "OpenPCDet frontend hook profile for inference export. "
            "'safe' only applies conservative cleanup; "
            "'denoise' applies voxel-neighborhood denoising; "
            "'adaptive' applies safe cleanup + intensity auto-repair, with optional guarded denoise; "
            "'adaptive_enhance' applies adaptive + distance-adaptive voxel-stat foreground enhancement; "
            "'adaptive_ground' applies adaptive + sparse near-ground denoise; "
            "'temporal_3frame' stacks neighbor frames with optional pose compensation."
        ),
    )
    parser.add_argument(
        "--openpcdet-frontend-xyz-clip-abs",
        type=float,
        default=1e4,
        help="Absolute xyz clip bound for OpenPCDet safe frontend profile.",
    )
    parser.add_argument(
        "--openpcdet-frontend-intensity-clip-abs",
        type=float,
        default=1e3,
        help="Absolute intensity clip bound for OpenPCDet safe frontend profile.",
    )
    parser.add_argument(
        "--openpcdet-frontend-denoise-voxel-size",
        type=float,
        default=0.25,
        help="Voxel size (meters) for OpenPCDet denoise frontend profile.",
    )
    parser.add_argument(
        "--openpcdet-frontend-denoise-neighbor-radius-voxels",
        type=int,
        default=1,
        help="Neighbor search radius in voxels for OpenPCDet denoise frontend profile.",
    )
    parser.add_argument(
        "--openpcdet-frontend-denoise-min-neighbor-points",
        type=int,
        default=4,
        help="Minimum neighboring points to keep points in denoise profile.",
    )
    parser.add_argument(
        "--openpcdet-frontend-temporal-pose-file",
        default="",
        help="Optional pose file for temporal_3frame frontend profile.",
    )
    parser.add_argument(
        "--openpcdet-frontend-temporal-allow-missing-pose",
        action="store_true",
        help=(
            "Allow temporal neighbor frames when pose is missing "
            "(fallback to identity transform)."
        ),
    )
    parser.add_argument(
        "--openpcdet-frontend-adaptive-enable-denoise",
        action="store_true",
        help="Enable guarded denoise in adaptive profile.",
    )
    parser.add_argument(
        "--openpcdet-frontend-adaptive-enable-feature-enhance",
        action="store_true",
        help="Enable distance-adaptive voxel-stat feature enhancement in adaptive profile.",
    )
    parser.add_argument(
        "--openpcdet-frontend-adaptive-max-denoise-drop-ratio",
        type=float,
        default=0.02,
        help="Max drop ratio to apply denoise in adaptive profile.",
    )
    parser.add_argument(
        "--openpcdet-frontend-adaptive-feature-grid-size-xy",
        type=float,
        default=0.8,
        help="Base XY voxel size (meters) for distance-adaptive feature enhancement.",
    )
    parser.add_argument(
        "--openpcdet-frontend-adaptive-feature-min-relative-height",
        type=float,
        default=0.2,
        help="Minimum local relative height (meters) to start foreground boosting.",
    )
    parser.add_argument(
        "--openpcdet-frontend-adaptive-feature-relative-height-scale",
        type=float,
        default=1.2,
        help="Relative height scale (meters) for foreground boosting saturation.",
    )
    parser.add_argument(
        "--openpcdet-frontend-adaptive-feature-boost-strength",
        type=float,
        default=0.2,
        help="Foreground intensity boost strength for feature enhancement.",
    )
    parser.add_argument(
        "--openpcdet-frontend-adaptive-feature-ground-suppress-strength",
        type=float,
        default=0.03,
        help="Ground intensity suppression strength for feature enhancement.",
    )
    parser.add_argument(
        "--openpcdet-frontend-adaptive-feature-max-adjust-ratio",
        type=float,
        default=0.35,
        help="If adjusted-point ratio is above this value, feature enhancement is skipped.",
    )
    parser.add_argument(
        "--openpcdet-frontend-adaptive-intensity-lower-percentile",
        type=float,
        default=1.0,
        help="Lower percentile for adaptive intensity auto-repair.",
    )
    parser.add_argument(
        "--openpcdet-frontend-adaptive-intensity-upper-percentile",
        type=float,
        default=99.0,
        help="Upper percentile for adaptive intensity auto-repair.",
    )
    parser.add_argument(
        "--openpcdet-frontend-adaptive-intensity-trigger-low",
        type=float,
        default=-0.1,
        help="Lower trigger bound for adaptive intensity auto-repair.",
    )
    parser.add_argument(
        "--openpcdet-frontend-adaptive-intensity-trigger-high",
        type=float,
        default=1.5,
        help="Upper trigger bound for adaptive intensity auto-repair.",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_kitti_dirs(kitti_root: Path) -> Dict[str, Path]:
    bin_dir = kitti_root / "training" / "velodyne"
    calib_dir = kitti_root / "training" / "calib"
    label_dir = kitti_root / "training" / "label_2"

    missing = [p for p in (bin_dir, calib_dir, label_dir) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required KITTI directories: "
            + ", ".join(str(p) for p in missing)
        )

    return {
        "bin_dir": bin_dir,
        "calib_dir": calib_dir,
        "label_dir": label_dir,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    kitti_root = Path(args.kitti_root).expanduser().resolve()
    kitti_dirs = ensure_kitti_dirs(kitti_root)

    workspace_dir = Path(args.workspace_dir).expanduser().resolve()
    pred_dir = workspace_dir / "predictions" / "data"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    exe_path = "cpp/evaluate_object.exe" if platform.system().lower().startswith("win") else "cpp/evaluate_object"
    python_bin = sys.executable
    t0 = time.time()

    print("\n[step 1/2] Export KITTI predictions with OpenPCDet")
    openpcdet_root = Path(args.openpcdet_root).expanduser().resolve()
    openpcdet_tools = openpcdet_root / "tools"
    openpcdet_cwd = openpcdet_tools if openpcdet_tools.exists() else openpcdet_root
    export_cmd = [
        python_bin,
        str(repo_root / "openpcdet_infer_kitti.py"),
        "--openpcdet-root",
        str(openpcdet_root),
        "--cfg-file",
        str(Path(args.openpcdet_cfg_file).expanduser().resolve()),
        "--ckpt",
        str(Path(args.openpcdet_ckpt).expanduser().resolve()),
        "--bin-dir",
        str(kitti_dirs["bin_dir"]),
        "--calib-dir",
        str(kitti_dirs["calib_dir"]),
        "--out-dir",
        str(pred_dir),
        "--device",
        args.openpcdet_device,
        "--frontend-profile",
        args.openpcdet_frontend_profile,
        "--frontend-xyz-clip-abs",
        str(args.openpcdet_frontend_xyz_clip_abs),
        "--frontend-intensity-clip-abs",
        str(args.openpcdet_frontend_intensity_clip_abs),
        "--frontend-denoise-voxel-size",
        str(args.openpcdet_frontend_denoise_voxel_size),
        "--frontend-denoise-neighbor-radius-voxels",
        str(args.openpcdet_frontend_denoise_neighbor_radius_voxels),
        "--frontend-denoise-min-neighbor-points",
        str(args.openpcdet_frontend_denoise_min_neighbor_points),
        "--frontend-adaptive-max-denoise-drop-ratio",
        str(args.openpcdet_frontend_adaptive_max_denoise_drop_ratio),
        "--frontend-adaptive-intensity-lower-percentile",
        str(args.openpcdet_frontend_adaptive_intensity_lower_percentile),
        "--frontend-adaptive-intensity-upper-percentile",
        str(args.openpcdet_frontend_adaptive_intensity_upper_percentile),
        "--frontend-adaptive-intensity-trigger-low",
        str(args.openpcdet_frontend_adaptive_intensity_trigger_low),
        "--frontend-adaptive-intensity-trigger-high",
        str(args.openpcdet_frontend_adaptive_intensity_trigger_high),
        "--frontend-adaptive-feature-grid-size-xy",
        str(args.openpcdet_frontend_adaptive_feature_grid_size_xy),
        "--frontend-adaptive-feature-min-relative-height",
        str(args.openpcdet_frontend_adaptive_feature_min_relative_height),
        "--frontend-adaptive-feature-relative-height-scale",
        str(args.openpcdet_frontend_adaptive_feature_relative_height_scale),
        "--frontend-adaptive-feature-boost-strength",
        str(args.openpcdet_frontend_adaptive_feature_boost_strength),
        "--frontend-adaptive-feature-ground-suppress-strength",
        str(args.openpcdet_frontend_adaptive_feature_ground_suppress_strength),
        "--frontend-adaptive-feature-max-adjust-ratio",
        str(args.openpcdet_frontend_adaptive_feature_max_adjust_ratio),
    ]
    if args.openpcdet_frontend_adaptive_enable_denoise:
        export_cmd.append("--frontend-adaptive-enable-denoise")
    if args.openpcdet_frontend_adaptive_enable_feature_enhance:
        export_cmd.append("--frontend-adaptive-enable-feature-enhance")
    if args.openpcdet_frontend_temporal_allow_missing_pose:
        export_cmd.append("--frontend-temporal-allow-missing-pose")
    if str(args.openpcdet_frontend_temporal_pose_file).strip():
        export_cmd.extend(
            [
                "--frontend-temporal-pose-file",
                str(Path(args.openpcdet_frontend_temporal_pose_file).expanduser().resolve()),
            ]
        )
    if args.limit > 0:
        export_cmd.extend(["--limit", str(args.limit)])
    run_cmd(export_cmd, openpcdet_cwd)

    print("\n[step 2/2] Run KITTI devkit evaluation")
    eval_cmd = [
        python_bin,
        "kitti_devkit_eval.py",
        "--label-dir",
        str(kitti_dirs["label_dir"]),
        "--pred-dir",
        str(pred_dir),
        "--run-name",
        args.run_name,
        "--compile",
        "--exe-path",
        exe_path,
    ]
    if platform.system().lower().startswith("win"):
        eval_cmd.extend(["--bash-path", args.bash_path])
    if args.limit > 0:
        eval_cmd.extend(["--n-test-images", str(args.limit)])
    run_cmd(eval_cmd, repo_root)

    dt = time.time() - t0
    final_summary = {
        "kitti_root": str(kitti_root),
        "workspace_dir": str(workspace_dir),
        "run_name": args.run_name,
        "limit": args.limit,
        "openpcdet_root": args.openpcdet_root,
        "openpcdet_cfg_file": args.openpcdet_cfg_file,
        "openpcdet_ckpt": args.openpcdet_ckpt,
        "openpcdet_device": args.openpcdet_device,
        "openpcdet_frontend_profile": args.openpcdet_frontend_profile,
        "openpcdet_frontend_xyz_clip_abs": args.openpcdet_frontend_xyz_clip_abs,
        "openpcdet_frontend_intensity_clip_abs": args.openpcdet_frontend_intensity_clip_abs,
        "openpcdet_frontend_denoise_voxel_size": args.openpcdet_frontend_denoise_voxel_size,
        "openpcdet_frontend_denoise_neighbor_radius_voxels": args.openpcdet_frontend_denoise_neighbor_radius_voxels,
        "openpcdet_frontend_denoise_min_neighbor_points": args.openpcdet_frontend_denoise_min_neighbor_points,
        "openpcdet_frontend_adaptive_enable_denoise": args.openpcdet_frontend_adaptive_enable_denoise,
        "openpcdet_frontend_adaptive_enable_feature_enhance": args.openpcdet_frontend_adaptive_enable_feature_enhance,
        "openpcdet_frontend_adaptive_max_denoise_drop_ratio": args.openpcdet_frontend_adaptive_max_denoise_drop_ratio,
        "openpcdet_frontend_adaptive_feature_grid_size_xy": args.openpcdet_frontend_adaptive_feature_grid_size_xy,
        "openpcdet_frontend_adaptive_feature_min_relative_height": args.openpcdet_frontend_adaptive_feature_min_relative_height,
        "openpcdet_frontend_adaptive_feature_relative_height_scale": args.openpcdet_frontend_adaptive_feature_relative_height_scale,
        "openpcdet_frontend_adaptive_feature_boost_strength": args.openpcdet_frontend_adaptive_feature_boost_strength,
        "openpcdet_frontend_adaptive_feature_ground_suppress_strength": args.openpcdet_frontend_adaptive_feature_ground_suppress_strength,
        "openpcdet_frontend_adaptive_feature_max_adjust_ratio": args.openpcdet_frontend_adaptive_feature_max_adjust_ratio,
        "openpcdet_frontend_adaptive_intensity_lower_percentile": args.openpcdet_frontend_adaptive_intensity_lower_percentile,
        "openpcdet_frontend_adaptive_intensity_upper_percentile": args.openpcdet_frontend_adaptive_intensity_upper_percentile,
        "openpcdet_frontend_adaptive_intensity_trigger_low": args.openpcdet_frontend_adaptive_intensity_trigger_low,
        "openpcdet_frontend_adaptive_intensity_trigger_high": args.openpcdet_frontend_adaptive_intensity_trigger_high,
        "openpcdet_frontend_temporal_pose_file": args.openpcdet_frontend_temporal_pose_file,
        "openpcdet_frontend_temporal_allow_missing_pose": args.openpcdet_frontend_temporal_allow_missing_pose,
        "pred_dir": str(pred_dir),
        "eval_summary_json": str(repo_root / "results" / args.run_name / "kitti_eval_summary.json"),
        "elapsed_sec": round(dt, 2),
    }
    final_path = workspace_dir / "full_pipeline_summary.json"
    final_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

    print("\n[done] OpenPCDet-only KITTI pipeline completed.")
    print(f"[saved] {final_path}")


if __name__ == "__main__":
    main()
