import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run KITTI C++ devkit evaluation against exported predictions and "
            "write a compact AP summary JSON."
        )
    )
    parser.add_argument(
        "--label-dir",
        required=True,
        help="Path to KITTI ground-truth label_2 directory.",
    )
    parser.add_argument(
        "--pred-dir",
        default="kitti_predictions/data",
        help="Path to prediction .txt files (KITTI format).",
    )
    parser.add_argument(
        "--run-name",
        default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Subfolder name under results/ used by the C++ devkit.",
    )
    parser.add_argument(
        "--n-test-images",
        type=int,
        default=7518,
        help=(
            "Number of indexed frame files required by this devkit build "
            "(default matches evaluate_object.cpp constant)."
        ),
    )
    parser.add_argument(
        "--bash-path",
        default=r"C:\msys64\usr\bin\bash.exe",
        help="Path to MSYS2 bash.exe used to compile/run the evaluator on Windows.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile cpp/evaluate_object.cpp before running evaluation.",
    )
    parser.add_argument(
        "--exe-path",
        default="cpp/evaluate_object.exe",
        help="Path to evaluator executable.",
    )
    return parser.parse_args()


def to_msys_path(path: Path) -> str:
    abs_path = path.resolve()
    raw = str(abs_path).replace("\\", "/")
    if len(raw) >= 2 and raw[1] == ":":
        drive = raw[0].lower()
        return f"/{drive}{raw[2:]}"
    return raw


def run_bash(command: str, bash_path: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [bash_path, "-lc", command],
        cwd=str(cwd),
        capture_output=True,
        text=False,
        check=False,
    )


def decode_stream(payload: bytes) -> str:
    if not payload:
        return ""
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def compile_devkit(repo_root: Path, bash_path: str) -> None:
    msys_root = to_msys_path(repo_root)
    command = (
        "export PATH=/ucrt64/bin:$PATH; "
        f"cd {msys_root} && "
        "/ucrt64/bin/g++ cpp/evaluate_object.cpp -o cpp/evaluate_object.exe -O2 -std=c++17"
    )
    result = run_bash(command, bash_path, repo_root)
    if result.returncode != 0:
        raise RuntimeError(
            "Devkit compile failed.\n"
            f"stdout:\n{decode_stream(result.stdout)}\n"
            f"stderr:\n{decode_stream(result.stderr)}"
        )


def discover_indexed_txt_files(directory: Path) -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    for file_path in sorted(directory.glob("*.txt")):
        stem = file_path.stem
        if stem.isdigit():
            mapping[int(stem)] = file_path
    return mapping


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def touch_empty(path: Path) -> None:
    ensure_parent(path)
    if path.exists() or path.is_symlink():
        path.unlink()
    path.write_text("", encoding="utf-8")


def prepare_devkit_layout(
    repo_root: Path,
    label_dir: Path,
    pred_dir: Path,
    run_name: str,
    n_test_images: int,
) -> Dict[str, int]:
    gt_target_dir = repo_root / "data" / "object" / "label_2"
    result_data_dir = repo_root / "results" / run_name / "data"
    gt_target_dir.mkdir(parents=True, exist_ok=True)
    result_data_dir.mkdir(parents=True, exist_ok=True)

    label_map = discover_indexed_txt_files(label_dir)
    pred_map = discover_indexed_txt_files(pred_dir)

    gt_copied = 0
    gt_empty = 0
    pred_copied = 0
    pred_empty = 0

    for idx in range(n_test_images):
        frame_name = f"{idx:06d}.txt"

        gt_dst = gt_target_dir / frame_name
        gt_src = label_map.get(idx)
        if gt_src is not None:
            link_or_copy(gt_src, gt_dst)
            gt_copied += 1
        else:
            touch_empty(gt_dst)
            gt_empty += 1

        pred_dst = result_data_dir / frame_name
        pred_src = pred_map.get(idx)
        if pred_src is not None:
            shutil.copy2(pred_src, pred_dst)
            pred_copied += 1
        else:
            touch_empty(pred_dst)
            pred_empty += 1

    return {
        "gt_copied": gt_copied,
        "gt_empty": gt_empty,
        "pred_copied": pred_copied,
        "pred_empty": pred_empty,
        "labels_found": len(label_map),
        "predictions_found": len(pred_map),
    }


def parse_stats_file(path: Path) -> Dict[str, float]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3:
        return {}

    difficulties = ["easy", "moderate", "hard"]
    summary: Dict[str, float] = {}
    for diff_name, line in zip(difficulties, lines[:3]):
        values = [float(x) for x in line.split()]
        if values:
            summary[diff_name] = sum(values) / len(values) * 100.0
    return summary


def collect_ap_summary(result_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    classes = ["car", "pedestrian", "cyclist"]
    metrics = {
        "2d": "detection",
        "bev": "detection_ground",
        "3d": "detection_3d",
    }

    output: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cls in classes:
        cls_metrics: Dict[str, Dict[str, float]] = {}
        for metric_name, suffix in metrics.items():
            file_path = result_dir / f"stats_{cls}_{suffix}.txt"
            if file_path.exists():
                cls_metrics[metric_name] = parse_stats_file(file_path)
        if cls_metrics:
            output[cls] = cls_metrics
    return output


def format_console_summary(ap_summary: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    lines: List[str] = []
    for cls_name in ["car", "pedestrian", "cyclist"]:
        cls_data = ap_summary.get(cls_name, {})
        if not cls_data:
            continue
        lines.append(f"[{cls_name}]")
        for metric in ["2d", "bev", "3d"]:
            metric_data = cls_data.get(metric)
            if not metric_data:
                continue
            easy = metric_data.get("easy", float("nan"))
            mod = metric_data.get("moderate", float("nan"))
            hard = metric_data.get("hard", float("nan"))
            lines.append(
                f"  {metric}: easy={easy:.2f} moderate={mod:.2f} hard={hard:.2f}"
            )
    return "\n".join(lines)


def run_evaluation(
    repo_root: Path,
    run_name: str,
    bash_path: str,
    exe_path: Path,
    n_test_images: int,
) -> Tuple[str, str]:
    if not exe_path.exists():
        raise FileNotFoundError(f"Evaluator not found: {exe_path}")

    msys_root = to_msys_path(repo_root)
    command = (
        "export PATH=/ucrt64/bin:$PATH; "
        f"cd {msys_root} && "
        f"./{exe_path.as_posix()} {run_name} {n_test_images}"
    )
    result = run_bash(command, bash_path, repo_root)
    if result.returncode != 0:
        raise RuntimeError(
            "Devkit evaluation failed.\n"
            f"stdout:\n{decode_stream(result.stdout)}\n"
            f"stderr:\n{decode_stream(result.stderr)}"
        )
    return decode_stream(result.stdout), decode_stream(result.stderr)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    label_dir = Path(args.label_dir)
    pred_dir = Path(args.pred_dir)
    exe_path = Path(args.exe_path)

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    if args.compile:
        print("[step] Compiling KITTI devkit evaluator...")
        compile_devkit(repo_root, args.bash_path)
        print(f"[ok] Built evaluator: {exe_path}")

    print("[step] Preparing KITTI devkit folder layout...")
    prep = prepare_devkit_layout(
        repo_root=repo_root,
        label_dir=label_dir,
        pred_dir=pred_dir,
        run_name=args.run_name,
        n_test_images=args.n_test_images,
    )
    print(
        "[ok] Prepared files: "
        f"gt_copied={prep['gt_copied']} gt_empty={prep['gt_empty']} "
        f"pred_copied={prep['pred_copied']} pred_empty={prep['pred_empty']}"
    )

    print("[step] Running C++ KITTI devkit evaluation...")
    stdout, stderr = run_evaluation(
        repo_root=repo_root,
        run_name=args.run_name,
        bash_path=args.bash_path,
        exe_path=exe_path,
        n_test_images=args.n_test_images,
    )
    if stdout.strip():
        print(stdout.strip())
    if stderr.strip():
        print(stderr.strip())

    result_dir = repo_root / "results" / args.run_name
    ap_summary = collect_ap_summary(result_dir)

    payload = {
        "run_name": args.run_name,
        "label_dir": str(label_dir.resolve()),
        "pred_dir": str(pred_dir.resolve()),
        "n_test_images": args.n_test_images,
        "prepare_summary": prep,
        "ap_summary": ap_summary,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_json = result_dir / "kitti_eval_summary.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("[done] KITTI devkit evaluation complete.")
    print(f"[saved] {out_json}")
    pretty = format_console_summary(ap_summary)
    if pretty:
        print(pretty)
    else:
        print("[warn] No stats files were parsed. Check evaluator output.")


if __name__ == "__main__":
    main()
