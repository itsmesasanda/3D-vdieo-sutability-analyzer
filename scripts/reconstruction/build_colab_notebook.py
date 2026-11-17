#!/usr/bin/env python3
"""Build the checked-in, syntax-validated Milestone 3B Colab notebook."""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path


OUTPUT = Path("notebooks/colmap_reconstruction_colab.ipynb")


def source(text: str) -> list[str]:
    value = textwrap.dedent(text).strip("\n") + "\n"
    return value.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str) -> dict:
    value = textwrap.dedent(text).strip("\n") + "\n"
    ast.parse(value)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": value.splitlines(keepends=True),
    }


CELLS = [
    markdown(
        """
        # Milestone 3B — sparse COLMAP reconstruction in Google Colab Free

        This notebook runs **sparse reconstruction only** on already extracted,
        timestamped frames. It does not download CO3D, train a model, change v1/v2
        scoring, run dense reconstruction, or claim that registration proves object
        completeness.

        The default run processes only `IMG_3913` as the pilot. Every COLMAP command,
        log, database, sparse model, text model, PLY, pose CSV, figure, environment
        snapshot, and summary row is copied to Google Drive because the Colab VM is
        temporary. Failed reconstruction is valid evidence; unavailable values remain
        blank.

        Before running, follow the preparation instructions in the final notebook
        cell or `evaluation/reconstruction/COLAB_LINUX_RUNBOOK.md`.
        """
    ),
    code(
        """
        #@title 1. Configuration { display-mode: "form" }
        from datetime import datetime, timezone
        from pathlib import Path

        INPUT_MODE = "drive"  #@param ["drive", "upload_zip"]
        DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/3d-video-suitability-analyzer"  #@param {type:"string"}
        DRIVE_INPUT_ROOT = "/content/drive/MyDrive/3d-video-suitability-analyzer/evaluation/reconstruction"  #@param {type:"string"}
        DRIVE_OUTPUT_PARENT = "/content/drive/MyDrive/3d-video-suitability-results"  #@param {type:"string"}
        DEVICE_PREFERENCE = "auto"  #@param ["auto", "cpu", "gpu"]
        PILOT_SEQUENCE = "IMG_3913"
        EXPECTED_PILOT_FRAME_COUNT = 35

        # Safety gates: leave both False for the first Run all.
        TRY_CONSERVATIVE_FALLBACK = False  #@param {type:"boolean"}
        PROCESS_REMAINING_AFTER_PILOT = False  #@param {type:"boolean"}
        REMAINING_SEQUENCES = ["IMG_3914", "IMG_3915", "orbit_good"]

        if "RUN_ID" not in globals():
            RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        print("Run ID:", RUN_ID)
        print("Input mode:", INPUT_MODE)
        print("Pilot only by default; fallback and remaining-video gates are disabled.")
        """
    ),
    code(
        """
        #@title 2. Mount Drive, select input, and locate project source
        import json
        import os
        import shutil
        import zipfile
        from google.colab import drive, files

        drive.mount("/content/drive")
        PERSISTENT_RUN_ROOT = Path(DRIVE_OUTPUT_PARENT) / f"milestone3b_{RUN_ID}"
        PERSISTENT_RUN_ROOT.mkdir(parents=True, exist_ok=False)
        (PERSISTENT_RUN_ROOT / "setup_logs").mkdir()

        def safe_extract(zip_path: Path, destination: Path) -> None:
            destination.mkdir(parents=True, exist_ok=True)
            root = destination.resolve()
            with zipfile.ZipFile(zip_path) as archive:
                for member in archive.infolist():
                    target = (destination / member.filename).resolve()
                    if root not in target.parents and target != root:
                        raise RuntimeError(f"Unsafe ZIP member path: {member.filename}")
                archive.extractall(destination)

        if INPUT_MODE == "upload_zip":
            print("Upload one ZIP containing the repository source and IMG_3913 frame directory.")
            uploaded = files.upload()
            zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
            if len(zip_names) != 1:
                raise RuntimeError("Upload exactly one .zip file")
            upload_path = Path("/content") / Path(zip_names[0]).name
            upload_path.write_bytes(uploaded[zip_names[0]])
            INPUT_ROOT = Path("/content/colab_input")
            safe_extract(upload_path, INPUT_ROOT)
        elif INPUT_MODE == "drive":
            INPUT_ROOT = Path(DRIVE_INPUT_ROOT)
            if not INPUT_ROOT.exists():
                raise FileNotFoundError(f"Drive input root does not exist: {INPUT_ROOT}")
        else:
            raise ValueError(f"Unsupported INPUT_MODE: {INPUT_MODE}")

        project_candidates = [Path(DRIVE_PROJECT_ROOT), INPUT_ROOT]
        PROJECT_ROOT = next(
            (
                candidate
                for candidate in project_candidates
                if (candidate / "src/video_suitability/reconstruction.py").is_file()
            ),
            None,
        )
        if PROJECT_ROOT is None:
            matches = list(INPUT_ROOT.rglob("src/video_suitability/reconstruction.py"))
            if matches:
                PROJECT_ROOT = matches[0].parents[2]
        if PROJECT_ROOT is None:
            raise FileNotFoundError(
                "Project source was not found. Upload/mount a root containing "
                "src/video_suitability/reconstruction.py."
            )

        configuration = {
            "run_id": RUN_ID,
            "input_mode": INPUT_MODE,
            "input_root": str(INPUT_ROOT),
            "project_root": str(PROJECT_ROOT),
            "persistent_run_root": str(PERSISTENT_RUN_ROOT),
            "device_preference": DEVICE_PREFERENCE,
            "pilot_sequence": PILOT_SEQUENCE,
            "try_conservative_fallback": TRY_CONSERVATIVE_FALLBACK,
            "process_remaining_after_pilot": PROCESS_REMAINING_AFTER_PILOT,
        }
        (PERSISTENT_RUN_ROOT / "run_configuration.json").write_text(
            json.dumps(configuration, indent=2) + "\\n", encoding="utf-8"
        )
        print(json.dumps(configuration, indent=2))
        """
    ),
    code(
        """
        #@title 3. Install COLMAP, ffmpeg, and Python dependencies (logs persist to Drive)
        import subprocess
        import sys
        import time

        def run_setup(command, log_name):
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            log_path = PERSISTENT_RUN_ROOT / "setup_logs" / log_name
            log_path.write_text(
                "COMMAND: " + " ".join(command) + "\\n\\n" + (completed.stdout or ""),
                encoding="utf-8",
            )
            print(f"{log_name}: exit={completed.returncode}, runtime={time.perf_counter()-started:.2f}s")
            if completed.returncode != 0:
                print((completed.stdout or "")[-4000:])
                raise RuntimeError(f"Setup command failed; inspect {log_path}")

        run_setup(["apt-get", "update", "-qq"], "apt_update.log")
        run_setup(
            ["apt-get", "install", "-y", "colmap", "ffmpeg"],
            "apt_install_colmap_ffmpeg.log",
        )
        requirements = PROJECT_ROOT / "requirements.txt"
        if not requirements.is_file():
            raise FileNotFoundError(f"Missing dependency file: {requirements}")
        run_setup(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)],
            "pip_install_requirements.log",
        )
        print("Installation complete. No dataset was downloaded.")
        """
    ),
    code(
        """
        #@title 4. Verify environment, COLMAP version/flags, memory, disk, and selected device
        import platform
        import re

        def capture(command):
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            return completed.returncode, completed.stdout or ""

        colmap_path = shutil.which("colmap")
        ffmpeg_path = shutil.which("ffmpeg")
        if not colmap_path or not ffmpeg_path:
            raise RuntimeError("COLMAP or ffmpeg is still unavailable after installation")

        _, colmap_help = capture([colmap_path, "-h"])
        _, feature_help = capture([colmap_path, "feature_extractor", "-h"])
        _, sequential_help = capture([colmap_path, "sequential_matcher", "-h"])
        _, exhaustive_help = capture([colmap_path, "exhaustive_matcher", "-h"])
        _, ffmpeg_version = capture([ffmpeg_path, "-version"])
        version_match = re.search(r"COLMAP\s+([0-9][^\s]*)", colmap_help, re.IGNORECASE)
        colmap_version = version_match.group(1) if version_match else colmap_help.splitlines()[0]

        feature_gpu_flag = next(
            (flag for flag in ["--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"] if flag in feature_help),
            None,
        )
        matching_gpu_flag = next(
            (flag for flag in ["--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"] if flag in sequential_help),
            None,
        )
        gpu_flags_supported = bool(feature_gpu_flag and matching_gpu_flag)
        nvidia_code, nvidia_output = capture(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        gpu_hardware_available = nvidia_code == 0 and bool(nvidia_output.strip())
        gpu_execution_verified = False
        if DEVICE_PREFERENCE != "cpu" and gpu_hardware_available and gpu_flags_supported:
            # Help flags alone do not prove that the apt binary was CUDA-enabled.
            # Verify extraction and matching on copies of two pilot frames.
            image_candidates = list(INPUT_ROOT.rglob(f"{PILOT_SEQUENCE}/images/*.jpg"))
            if not image_candidates:
                image_candidates = list((PROJECT_ROOT / "evaluation/reconstruction" / PILOT_SEQUENCE / "images").glob("*.jpg"))
            if len(image_candidates) >= 2:
                smoke_root = Path("/content/gpu_smoke") / RUN_ID
                smoke_images = smoke_root / "images"
                smoke_images.mkdir(parents=True, exist_ok=False)
                for path in sorted(image_candidates)[:2]:
                    shutil.copy2(path, smoke_images / path.name)
                smoke_database = smoke_root / "database.db"
                feature_code, feature_output = capture([
                    colmap_path, "feature_extractor", "--database_path", str(smoke_database),
                    "--image_path", str(smoke_images), "--ImageReader.single_camera", "1",
                    feature_gpu_flag, "1",
                ])
                match_code, match_output = (-1, "skipped because GPU feature extraction failed")
                if feature_code == 0:
                    match_code, match_output = capture([
                        colmap_path, "sequential_matcher", "--database_path", str(smoke_database),
                        "--SequentialMatching.overlap", "1", matching_gpu_flag, "1",
                    ])
                (PERSISTENT_RUN_ROOT / "setup_logs" / "gpu_feature_smoke.log").write_text(feature_output, encoding="utf-8")
                (PERSISTENT_RUN_ROOT / "setup_logs" / "gpu_matching_smoke.log").write_text(match_output, encoding="utf-8")
                gpu_execution_verified = feature_code == 0 and match_code == 0
        if DEVICE_PREFERENCE == "gpu" and not gpu_execution_verified:
            raise RuntimeError(
                "GPU was required, but hardware + flags + extraction/matching smoke verification did not all succeed"
            )
        USE_GPU = DEVICE_PREFERENCE != "cpu" and gpu_execution_verified
        selected_device = nvidia_output.strip() if USE_GPU else "CPU"

        disk = shutil.disk_usage("/content")
        meminfo = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            meminfo[key] = value.strip()
        environment = {
            "runtime_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python": sys.version,
            "logical_cpu_count": os.cpu_count(),
            "memory_total": meminfo.get("MemTotal"),
            "memory_available": meminfo.get("MemAvailable"),
            "disk_total_gib": round(disk.total / 1024**3, 3),
            "disk_free_gib": round(disk.free / 1024**3, 3),
            "colmap_path": colmap_path,
            "colmap_version": colmap_version,
            "ffmpeg_path": ffmpeg_path,
            "ffmpeg_version_line": ffmpeg_version.splitlines()[0],
            "feature_gpu_flag": feature_gpu_flag,
            "matching_gpu_flag": matching_gpu_flag,
            "gpu_hardware_available": gpu_hardware_available,
            "gpu_flags_supported": gpu_flags_supported,
            "gpu_execution_verified": gpu_execution_verified,
            "selected_device": selected_device,
        }
        (PERSISTENT_RUN_ROOT / "environment.json").write_text(
            json.dumps(environment, indent=2) + "\\n", encoding="utf-8"
        )
        for name, content in {
            "colmap_help.log": colmap_help,
            "feature_extractor_help.log": feature_help,
            "sequential_matcher_help.log": sequential_help,
            "exhaustive_matcher_help.log": exhaustive_help,
            "ffmpeg_version.log": ffmpeg_version,
            "nvidia_smi.log": nvidia_output,
        }.items():
            (PERSISTENT_RUN_ROOT / "setup_logs" / name).write_text(content, encoding="utf-8")
        print(json.dumps(environment, indent=2))
        """
    ),
    code(
        """
        #@title 5. Import the project module; verify frames, manifest hashes, and frozen scores
        import csv
        import sys
        from IPython.display import Image as DisplayImage, display

        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from video_suitability.reconstruction import run_sparse_colmap, sha256_file

        def find_sequence_directory(sequence):
            candidates = [
                INPUT_ROOT / sequence,
                INPUT_ROOT / "evaluation" / "reconstruction" / sequence,
                PROJECT_ROOT / "evaluation" / "reconstruction" / sequence,
            ]
            if INPUT_ROOT.name == sequence:
                candidates.insert(0, INPUT_ROOT)
            for candidate in candidates:
                if (candidate / "images").is_dir():
                    return candidate
            return None

        prediction_candidates = [
            PROJECT_ROOT / "evaluation/reconstruction/results.csv",
            INPUT_ROOT / "evaluation/reconstruction/results.csv",
            INPUT_ROOT / "results.csv",
        ]
        prediction_path = next((path for path in prediction_candidates if path.is_file()), None)
        if prediction_path is None:
            raise FileNotFoundError(
                "Preserved evaluation/reconstruction/results.csv is required; scores will not be recomputed."
            )
        with prediction_path.open(newline="", encoding="utf-8") as handle:
            PREDICTION_ROWS = list(csv.DictReader(handle))
        PREDICTIONS_BY_STEM = {Path(row["filename"]).stem: row for row in PREDICTION_ROWS}
        shutil.copy2(prediction_path, PERSISTENT_RUN_ROOT / "preserved_v1_v2_predictions.csv")

        def verify_sequence_input(sequence, expected_count=None):
            sequence_dir = find_sequence_directory(sequence)
            if sequence_dir is None:
                return None
            images_dir = sequence_dir / "images"
            images = sorted(images_dir.glob("*.jpg"))
            if len(images) < 2:
                raise RuntimeError(f"{sequence}: fewer than two JPEG frames")
            if expected_count is not None and len(images) != expected_count:
                raise RuntimeError(
                    f"{sequence}: expected {expected_count} extracted frames, found {len(images)}"
                )
            hashes = {image.name: sha256_file(image) for image in images}
            manifest = sequence_dir / "frames_manifest.csv"
            manifest_verified = False
            if manifest.is_file():
                with manifest.open(newline="", encoding="utf-8") as handle:
                    manifest_rows = list(csv.DictReader(handle))
                manifest_hashes = {row["filename"]: row["sha256"] for row in manifest_rows}
                if manifest_hashes != hashes:
                    raise RuntimeError(f"{sequence}: frame hashes differ from frames_manifest.csv")
                manifest_verified = True
            evidence = {
                "sequence": sequence,
                "sequence_directory": str(sequence_dir),
                "images_directory": str(images_dir),
                "frame_count": len(images),
                "manifest_verified": manifest_verified,
                "first_frame": images[0].name,
                "last_frame": images[-1].name,
                "frame_sha256": hashes,
            }
            output = PERSISTENT_RUN_ROOT / sequence / "input_integrity.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(evidence, indent=2) + "\\n", encoding="utf-8")
            return {"sequence_dir": sequence_dir, "images_dir": images_dir, "evidence": evidence}

        SEQUENCE_INPUTS = {}
        pilot_input = verify_sequence_input(PILOT_SEQUENCE, EXPECTED_PILOT_FRAME_COUNT)
        if pilot_input is None:
            raise FileNotFoundError(f"Could not find {PILOT_SEQUENCE}/images under configured input roots")
        SEQUENCE_INPUTS[PILOT_SEQUENCE] = pilot_input
        pilot_prediction = PREDICTIONS_BY_STEM.get(PILOT_SEQUENCE)
        if pilot_prediction is None:
            raise RuntimeError(f"No preserved v1/v2 prediction row for {PILOT_SEQUENCE}")
        print(
            f"Verified {pilot_input['evidence']['frame_count']} immutable pilot frames; "
            f"manifest_verified={pilot_input['evidence']['manifest_verified']}"
        )
        print(
            "Preserved prediction — v1:", pilot_prediction["baseline_v1_score"],
            "v2:", pilot_prediction["v2_score"], "(not recomputed)"
        )
        display(DisplayImage(filename=str(sorted(pilot_input["images_dir"].glob("*.jpg"))[0]), width=300))
        display(DisplayImage(filename=str(sorted(pilot_input["images_dir"].glob("*.jpg"))[-1]), width=300))
        """
    ),
    code(
        """
        #@title 6. Persistent experiment helpers
        import traceback

        RESULT_ROWS = []
        LOCAL_RUN_ROOT = Path("/content/colmap_work") / RUN_ID
        LOCAL_RUN_ROOT.mkdir(parents=True, exist_ok=False)

        SUMMARY_FIELDS = [
            "sequence", "attempt", "status", "failure_stage", "failure_reason",
            "baseline_v1_score", "v2_score", "selected_device", "camera_model",
            "total_input_image_count", "registered_image_count", "registered_image_ratio",
            "reconstructed_point_count", "mean_reprojection_error_pixels",
            "mean_track_length_images", "mapper_runtime_seconds", "total_runtime_seconds",
            "persistent_output_directory", "completeness_claimed",
        ]

        def rewrite_paths(value, old_root, new_root):
            if isinstance(value, str):
                return value.replace(str(old_root), str(new_root))
            if isinstance(value, list):
                return [rewrite_paths(item, old_root, new_root) for item in value]
            if isinstance(value, dict):
                return {key: rewrite_paths(item, old_root, new_root) for key, item in value.items()}
            return value

        def update_summary_csv():
            path = PERSISTENT_RUN_ROOT / "reconstruction_summary.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
                writer.writeheader()
                writer.writerows(RESULT_ROWS)
            return path

        def persist_camera_poses(result, persistent_workspace):
            metrics = result.get("objective_metrics") or {}
            poses = metrics.get("camera_centers") or []
            if not poses:
                return None
            pose_path = persistent_workspace / "camera_poses.csv"
            fields = [
                "image_id", "camera_id", "name", "qw", "qx", "qy", "qz",
                "tx", "ty", "tz", "center_x", "center_y", "center_z",
            ]
            with pose_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for pose in poses:
                    q = pose["qvec_world_to_camera_qw_qx_qy_qz"]
                    t = pose["tvec_world_to_camera_xyz"]
                    c = pose["xyz_arbitrary_units"]
                    writer.writerow({
                        "image_id": pose["image_id"], "camera_id": pose["camera_id"],
                        "name": pose["name"], "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3],
                        "tx": t[0], "ty": t[1], "tz": t[2],
                        "center_x": c[0], "center_y": c[1], "center_z": c[2],
                    })
            return pose_path

        def append_result_row(sequence, attempt, result, persistent_workspace):
            prediction = PREDICTIONS_BY_STEM.get(sequence, {})
            metrics = result.get("objective_metrics") or {}
            row = {
                "sequence": sequence,
                "attempt": attempt,
                "status": result.get("status"),
                "failure_stage": result.get("failure_stage") or "",
                "failure_reason": result.get("failure_reason") or "",
                "baseline_v1_score": prediction.get("baseline_v1_score", ""),
                "v2_score": prediction.get("v2_score", ""),
                "selected_device": result.get("selected_device", ""),
                "camera_model": result.get("camera_model", ""),
                "total_input_image_count": result.get("total_input_image_count", ""),
                "registered_image_count": metrics.get("registered_image_count", ""),
                "registered_image_ratio": metrics.get("registered_image_ratio", ""),
                "reconstructed_point_count": metrics.get("reconstructed_point_count", ""),
                "mean_reprojection_error_pixels": metrics.get("mean_reprojection_error_pixels", ""),
                "mean_track_length_images": metrics.get("mean_track_length_images", ""),
                "mapper_runtime_seconds": result.get("mapper_runtime_seconds", ""),
                "total_runtime_seconds": result.get("total_runtime_seconds", ""),
                "persistent_output_directory": str(persistent_workspace),
                "completeness_claimed": False,
            }
            RESULT_ROWS.append(row)
            update_summary_csv()
            return row

        def run_attempt(sequence, attempt, camera_model="OPENCV", overlap=10, retry_exhaustive=True, use_gpu=False):
            sequence_input = SEQUENCE_INPUTS[sequence]
            images_dir = sequence_input["images_dir"]
            before = sequence_input["evidence"]["frame_sha256"]
            local_workspace = LOCAL_RUN_ROOT / sequence / attempt
            persistent_workspace = PERSISTENT_RUN_ROOT / sequence / attempt
            started = time.perf_counter()
            try:
                result = run_sparse_colmap(
                    images_dir,
                    local_workspace,
                    colmap_executable=colmap_path,
                    camera_model=camera_model,
                    sequential_overlap=overlap,
                    retry_exhaustive=retry_exhaustive,
                    use_gpu=use_gpu,
                )
            except Exception as exc:
                local_workspace.mkdir(parents=True, exist_ok=True)
                (local_workspace / "logs").mkdir(exist_ok=True)
                (local_workspace / "logs" / "python_exception.log").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                result = {
                    "schema_version": "1.0", "analysis_type": "sparse_colmap_reconstruction",
                    "dense_reconstruction_run": False, "status": "pipeline_exception",
                    "failure_stage": "python_wrapper", "failure_reason": str(exc),
                    "selected_device": "gpu" if use_gpu else "cpu", "camera_model": camera_model,
                    "total_input_image_count": len(before), "objective_metrics": None,
                    "mapper_runtime_seconds": None,
                    "total_runtime_seconds": round(time.perf_counter() - started, 6),
                }
                (local_workspace / "reconstruction_result.json").write_text(
                    json.dumps(result, indent=2) + "\\n", encoding="utf-8"
                )
            after = {path.name: sha256_file(path) for path in sorted(images_dir.glob("*.jpg"))}
            if before != after:
                raise RuntimeError(f"{sequence}: input frame checksum changed during COLMAP run")
            shutil.copytree(local_workspace, persistent_workspace)
            persistent_result = rewrite_paths(result, local_workspace, persistent_workspace)
            persistent_result["input_frames_unchanged"] = True
            persistent_result["sparse_reconstruction_does_not_prove_completeness"] = True
            (persistent_workspace / "reconstruction_result_persistent.json").write_text(
                json.dumps(persistent_result, indent=2) + "\\n", encoding="utf-8"
            )
            persist_camera_poses(persistent_result, persistent_workspace)
            row = append_result_row(sequence, attempt, persistent_result, persistent_workspace)
            print(json.dumps(row, indent=2))
            return persistent_result

        def print_failure_diagnostics(result):
            print("Status:", result.get("status"))
            print("Failure stage:", result.get("failure_stage"))
            print("Failure reason:", result.get("failure_reason"))
            for command in result.get("commands", []):
                log_path = Path(command["log_path"])
                if log_path.is_file() and command.get("returncode") != 0:
                    print(f"\\n--- tail of {log_path.name} ---")
                    print(log_path.read_text(encoding="utf-8", errors="replace")[-5000:])
        """
    ),
    code(
        """
        #@title 7. Run the IMG_3913 pilot (primary settings)
        PILOT_PRIMARY_RESULT = run_attempt(
            PILOT_SEQUENCE,
            "primary_opencv_sequential",
            camera_model="OPENCV",
            overlap=10,
            retry_exhaustive=True,
            use_gpu=USE_GPU,
        )
        PILOT_FINAL_RESULT = PILOT_PRIMARY_RESULT
        if PILOT_PRIMARY_RESULT["status"] == "succeeded":
            metrics = PILOT_PRIMARY_RESULT["objective_metrics"]
            print(
                "Pilot sparse model:", metrics["registered_image_count"], "/",
                metrics["total_input_image_count"], "registered;",
                metrics["reconstructed_point_count"], "points;",
                metrics["mean_reprojection_error_pixels"], "px mean reprojection error;",
                metrics["mean_track_length_images"], "mean track length."
            )
            display(DisplayImage(filename=PILOT_PRIMARY_RESULT["sparse_preview"], width=900))
        else:
            print("Primary pilot did not succeed. Review diagnostics before enabling fallback.")
            print_failure_diagnostics(PILOT_PRIMARY_RESULT)
        """
    ),
    code(
        """
        #@title 8. Optional conservative fallback — runs only after diagnostics and explicit opt-in
        if PILOT_PRIMARY_RESULT["status"] != "succeeded":
            print_failure_diagnostics(PILOT_PRIMARY_RESULT)
            if TRY_CONSERVATIVE_FALLBACK:
                print("Starting separate CPU SIMPLE_RADIAL attempt; primary evidence remains preserved.")
                PILOT_FALLBACK_RESULT = run_attempt(
                    PILOT_SEQUENCE,
                    "fallback_cpu_simple_radial",
                    camera_model="SIMPLE_RADIAL",
                    overlap=20,
                    retry_exhaustive=True,
                    use_gpu=False,
                )
                if PILOT_FALLBACK_RESULT["status"] == "succeeded":
                    PILOT_FINAL_RESULT = PILOT_FALLBACK_RESULT
                    display(DisplayImage(filename=PILOT_FALLBACK_RESULT["sparse_preview"], width=900))
                else:
                    print_failure_diagnostics(PILOT_FALLBACK_RESULT)
            else:
                print("Fallback not run. Set TRY_CONSERVATIVE_FALLBACK=True only after reviewing the logs.")
        else:
            print("Primary pilot succeeded; fallback is neither needed nor run.")
        """
    ),
    code(
        """
        #@title 9. Optional independent remaining sequences — gated by pilot success
        if PROCESS_REMAINING_AFTER_PILOT:
            if PILOT_FINAL_RESULT["status"] != "succeeded":
                print("Remaining sequences blocked: pilot has no successful sparse model.")
            else:
                for sequence in REMAINING_SEQUENCES:
                    sequence_input = verify_sequence_input(sequence)
                    if sequence_input is None:
                        print(f"{sequence}: input frames unavailable; skipping without fabricating measurements.")
                        prediction = PREDICTIONS_BY_STEM.get(sequence, {})
                        RESULT_ROWS.append({
                            "sequence": sequence, "attempt": "not_run", "status": "input_unavailable",
                            "failure_stage": "input", "failure_reason": "No isolated extracted-frame directory found",
                            "baseline_v1_score": prediction.get("baseline_v1_score", ""),
                            "v2_score": prediction.get("v2_score", ""), "selected_device": "",
                            "camera_model": "", "total_input_image_count": "", "registered_image_count": "",
                            "registered_image_ratio": "", "reconstructed_point_count": "",
                            "mean_reprojection_error_pixels": "", "mean_track_length_images": "",
                            "mapper_runtime_seconds": "", "total_runtime_seconds": "",
                            "persistent_output_directory": "", "completeness_claimed": False,
                        })
                        update_summary_csv()
                        continue
                    SEQUENCE_INPUTS[sequence] = sequence_input
                    result = run_attempt(
                        sequence, "primary_opencv_sequential", camera_model="OPENCV",
                        overlap=10, retry_exhaustive=True, use_gpu=USE_GPU,
                    )
                    if result["status"] != "succeeded":
                        print_failure_diagnostics(result)
        else:
            print("Remaining sequences not run. Enable only after reviewing a successful pilot.")
        """
    ),
    code(
        """
        #@title 10. Review persistent results
        import pandas as pd

        summary_path = update_summary_csv()
        summary = pd.read_csv(summary_path, keep_default_na=False)
        display(summary)
        print("Persistent results:", PERSISTENT_RUN_ROOT)
        print("Summary CSV:", summary_path)
        print(
            "Interpretation warning: registration, point count, reprojection error and track "
            "length are sparse-COLMAP measurements, not proof of object completeness or usability."
        )
        for row in RESULT_ROWS:
            if row["status"] == "succeeded":
                preview = Path(row["persistent_output_directory"]) / "sparse_preview.png"
                if preview.is_file():
                    print(row["sequence"], row["attempt"])
                    display(DisplayImage(filename=str(preview), width=900))
        """
    ),
    markdown(
        """
        ## Human action and result download

        1. On the local machine, prepare either:
           - **Drive mode:** upload this repository folder to
             `MyDrive/3d-video-suitability-analyzer`, including
             `src/`, `requirements.txt`, `evaluation/reconstruction/results.csv`, and
             `evaluation/reconstruction/IMG_3913/{images,frames_manifest.csv,extraction_metadata.json}`; or
           - **ZIP mode:** create one ZIP with those same relative paths, choose
             `upload_zip` in cell 1, and upload it when prompted.
        2. Open `notebooks/colmap_reconstruction_colab.ipynb` at
           [colab.research.google.com](https://colab.research.google.com/), select the
           free hosted runtime, and leave the accelerator on CPU unless the free
           runtime actually exposes a supported NVIDIA GPU.
        3. Run cells 1–7. Authorize Google Drive when prompted. Do not close the tab
           during installation or mapping.
        4. If the pilot fails, inspect the printed failure stage and Drive logs before
           setting `TRY_CONSERVATIVE_FALLBACK=True` and rerunning from cell 1. The
           fallback writes to a separate attempt directory.
        5. If the pilot succeeds and its sparse preview is plausible, add isolated
           frame directories for the other sequences, set
           `PROCESS_REMAINING_AFTER_PILOT=True`, and rerun in a **new run directory**.
        6. In Google Drive, open `3d-video-suitability-results/milestone3b_<UTC>/`.
           Download the entire folder or at minimum `reconstruction_summary.csv`,
           every `reconstruction_result_persistent.json`, `camera_poses.csv`,
           `sparse_points_and_cameras.ply`, `sparse_preview.png`, text models, and
           `logs/`.

        Colab Free availability and runtime limits are external constraints. A runtime
        disconnect does not imply a bad video; use the saved stage logs and status.
        Never interpret a sparse point cloud as a complete 3D object without separate
        human review of geometry, missing regions, and usability.
        """
    ),
]


def main() -> int:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "CPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

