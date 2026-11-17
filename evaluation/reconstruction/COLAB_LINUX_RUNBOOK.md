# Milestones 3A/3B sparse COLMAP runbook

This workflow runs **sparse** reconstruction only. It does not run dense stereo,
does not download a dataset, does not train a model, and does not change heuristic
v1/v2 reports. Run the pilot first. Only proceed to the other recordings when the
pilot produces a sparse model (`status: succeeded`).

## Why this workflow is needed

The audited macOS host has no `colmap`, `ffmpeg`, `ffprobe`, or `pycolmap` and only
1.820 GiB free at the final audit. Homebrew is present, but installing COLMAP's
large dependency tree plus retaining reconstruction intermediates is not safe at
that disk margin. The local OpenCV extraction path remains usable.

No cloud job has been started. Google Colab's free CPU runtime or an existing Linux
machine is sufficient, subject to that service's availability and storage limits.
Do not enable a paid runtime without explicit approval.

## Recommended Milestone 3B path: executable Colab notebook

Use `notebooks/colmap_reconstruction_colab.ipynb`. It installs the distribution's
COLMAP and ffmpeg packages, inspects actual CLI help before choosing flags, verifies
any proposed GPU path with a two-frame extraction/matching smoke test, and otherwise
runs CPU. CPU is sufficient for the 35-frame pilot.

### Prepare input using Google Drive

1. Create `MyDrive/3d-video-suitability-analyzer/`.
2. Copy this repository's `src/`, `requirements.txt`, and
   `evaluation/reconstruction/results.csv` into that folder with relative paths
   unchanged.
3. Copy the local pilot artifacts into:

   ```text
   MyDrive/3d-video-suitability-analyzer/evaluation/reconstruction/IMG_3913/
     images/                    # 35 original extracted JPEGs
     frames_manifest.csv
     extraction_metadata.json
   ```

4. Do not rename, recompress, rotate, or edit the JPEGs. The notebook verifies all
   manifest hashes before and after COLMAP.

The default notebook paths match this layout. Persistent results go to
`MyDrive/3d-video-suitability-results/milestone3b_<UTC>/`; they never replace the
local Milestone 3A evidence or v1/v2 reports.

### Alternative: prepare one uploaded ZIP

From the repository root on the local machine:

```bash
zip -r IMG_3913_colab_bundle.zip \
  notebooks/colmap_reconstruction_colab.ipynb \
  src requirements.txt \
  evaluation/reconstruction/results.csv \
  evaluation/reconstruction/IMG_3913/images \
  evaluation/reconstruction/IMG_3913/frames_manifest.csv \
  evaluation/reconstruction/IMG_3913/extraction_metadata.json
```

Set `INPUT_MODE = "upload_zip"` in cell 1. Google Drive is still mounted because all
results and logs must survive the temporary runtime.

### Open and run in Colab Free

1. Open <https://colab.research.google.com/> and choose **File → Upload notebook**.
2. Select `notebooks/colmap_reconstruction_colab.ipynb`.
3. Choose **Runtime → Change runtime type** and use the free CPU runtime initially;
   no paid accelerator is needed.
4. Leave `TRY_CONSERVATIVE_FALLBACK` and
   `PROCESS_REMAINING_AFTER_PILOT` set to `False`.
5. Run cells 1–7 in order and approve the Drive mount.
6. Inspect status, objective metrics, stage logs, and the sparse preview.
7. If the primary pilot fails, review diagnostics first. Only then enable the
   separate CPU `SIMPLE_RADIAL` fallback and run its cell. The primary workspace
   remains preserved.
8. Only after a successful, visually plausible pilot should the remaining-video
   gate be enabled. Each sequence needs its own isolated `images/` directory.

The notebook records COLMAP/ffmpeg versions, old/new GPU flag names, CPU/memory/disk
information, selected device, setup and stage logs, runtimes, registration metrics,
sparse points, reprojection error, track lengths, text models, PLY, camera poses, and
a preview. Missing measurements remain blank.

### Download and return results

Download the complete `milestone3b_<UTC>/` folder from Google Drive. At minimum,
retain `environment.json`, `run_configuration.json`, `setup_logs/`,
`reconstruction_summary.csv`, `preserved_v1_v2_predictions.csv`, every attempt's
`reconstruction_result_persistent.json` and `logs/`, `camera_poses.csv`,
`sparse_points_and_cameras.ply`, `sparse_preview.png`, text models, and sparse binary
models.

Return that folder for consolidation. Sparse registration is objective evidence,
but it does not establish object completeness, geometric plausibility, or practical
usability without human inspection.

## Linux/Colab setup

In a fresh Debian/Ubuntu or Colab shell:

```bash
sudo apt-get update
sudo apt-get install -y colmap ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
colmap -h
ffmpeg -version
```

For Colab, omit `sudo` if the image is already root. Copy or mount this repository
with the four source files in `data/samples/real_tests/`. Do not mix images from
different videos in one directory or one COLMAP database.

## 1. Audit and pilot extraction

```bash
python scripts/reconstruction/audit_environment.py \
  --out evaluation/reconstruction/environment_audit_linux.json

python scripts/reconstruction/extract_frames.py \
  --video data/samples/real_tests/IMG_3913.MOV \
  --output-dir evaluation/reconstruction/IMG_3913_linux \
  --sample-fps 2 \
  --max-side 1600 \
  --jpeg-quality 95
```

The extractor records the source checksum before and after, source frame indices,
timestamps, orientation metadata, output dimensions, per-frame checksums, and
runtime. It enables OpenCV orientation handling. Inspect several JPEGs before
mapping; they must be upright. The local pilot was visually checked and was upright
at 900×1600 despite a 90-degree container orientation tag.

## 2. Pilot sparse reconstruction

```bash
python scripts/reconstruction/run_colmap.py \
  --images-dir evaluation/reconstruction/IMG_3913_linux/images \
  --workspace-dir evaluation/reconstruction/IMG_3913_linux/colmap \
  --camera-model OPENCV \
  --sequential-overlap 10 \
  --retry-exhaustive
```

The runner selects the installed COLMAP version's CPU flags, shares one camera for
all frames (`ImageReader.single_camera=1`), starts with sequential matching, and
uses exhaustive matching only if requested and sequential mapping produces no
model. `OPENCV` provides a shared focal/principal-point/radial/tangential model for
these same-camera phone frames. If calibration is known independently, pass it in a
separate documented experiment; do not guess focal length from the heuristic.

The output is `reconstruction_result.json`. A successful run also contains COLMAP
logs, the selected text model (`cameras.txt`, `images.txt`, `points3D.txt`), a sparse
PLY, and `sparse_preview.png`. The preview shows sparse points and camera centers in
XY/XZ/YZ projections. It is not a mesh or dense reconstruction.

If the result is `reconstruction_failed` because sequential overlap is inadequate:

1. Preserve the failed workspace and logs.
2. Extract into a new directory at 3 FPS (never overwrite the 2 FPS evidence).
3. Rerun sequential matching there.
4. Use `--retry-exhaustive` only for the bounded sequence if needed.

Do not treat `infrastructure_unavailable` as reconstruction failure.

## 3. Gate before the remaining videos

Confirm that the pilot JSON says `"status": "succeeded"`, inspect the registration
ratio and sparse preview, and retain the pilot artifacts. Then run each sequence in
its own directory:

```bash
for name in IMG_3914 IMG_3915 orbit_good; do
  extension=MOV
  if [ "$name" = orbit_good ]; then extension=mp4; fi
  python scripts/reconstruction/extract_frames.py \
    --video "data/samples/real_tests/${name}.${extension}" \
    --output-dir "evaluation/reconstruction/${name}_linux" \
    --sample-fps 2 --max-side 1600 --jpeg-quality 95
  python scripts/reconstruction/run_colmap.py \
    --images-dir "evaluation/reconstruction/${name}_linux/images" \
    --workspace-dir "evaluation/reconstruction/${name}_linux/colmap" \
    --camera-model OPENCV --sequential-overlap 10 --retry-exhaustive
done
```

`orbit_good.mp4` is exactly 24 frames at 12 FPS and 2.0 seconds in the accessible
file. At 2 FPS it yields only about five frames, so retain the initial result but
also consider a separate all-frame experiment if the sparse mapper lacks support.
Record this as a cadence change, not as a hidden replacement.

## 4. Consolidate objective results

Copy the Linux/Colab result directories back without replacing the preserved local
pilot directory. Either point `build_evaluation.py` at a staged evaluation tree or
move reviewed Linux results into their final per-video directories after preserving
the local unavailable JSON under a distinct filename. Then run:

```bash
python scripts/reconstruction/build_evaluation.py
python -m pytest -q
```

The CSV leaves measurements blank when unavailable. Registration ratio, sparse
point count, reprojection error, track length, and camera-center extent are objective
COLMAP measurements. They are not ratings of object completeness or practical
usability; fill those only in `human_review_form.csv` after inspecting geometry.

