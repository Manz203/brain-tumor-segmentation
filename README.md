# NeuroSeg V2 — brain MRI segmentation research workspace

A patient-separated comparative study of U-Net and Attention U-Net for binary 2D MRI segmentation, with an inference dashboard and MLOps tooling. **Research and education only; not a clinically validated diagnostic system.**

## Start here: one Kaggle notebook

Open [Notebooks/03_neuroseg_end_to_end.ipynb](Notebooks/03_neuroseg_end_to_end.ipynb) in a private Kaggle notebook with GPU and Internet enabled. Run its single cell (Save & Run All to persist outputs). It will:

1. Recover the original MATLAB dataset from an attached private input or one of two hash-verified mirrors, or reuse a validated processed folder/ZIP.
2. Reproduce the patient-level split, normalize images and prepare binary masks.
3. Train U-Net and Attention U-Net (15 epochs maximum by default) with the same settings. Set TUNE=True to additionally compare learning rates; no test metrics are used for model selection.
4. Freeze the champion using best validation Dice. Evaluate that checkpoint once on the held-out test set. Reruns reuse a frozen selection instead of choosing based on test results.
5. Optionally track training histories and artifacts in MLflow and register the champion.
6. Export `/kaggle/working/neuroseg_submission.zip` with the selected weights, best candidate of each architecture, real evaluation CSV/JSON, figures and split manifest.

If MLflow dependency installation fails, set TRACK_MLFLOW=False to complete training first; tracking/registry then remains to be run. If both GitHub mirrors are inaccessible, attach the original MATLAB files as private input. No fake weights or metrics are shipped.

### CLI alternatives

`python -m neuroseg.study --epochs 15 --out neuroseg_results --mlflow`

Install `requirements-v2.txt` and optionally `requirements-mlops.txt`. Use `--tune` for two additional learning-rate candidates, `--multi-gpu` for optional DataParallel (not necessarily faster), or `--allow-cpu` only for development. The default processed data path is /tmp/neuroseg_processed; set --data to a writable location outside Kaggle if needed. The manifest and config checks protect comparability. Candidate checkpoints can resume in the same session; RNG state is not restored exactly.

For a single validation-only experiment: `python -m neuroseg.train --architecture attention_unet --skip-test --out attention_validation`. The study runner is the preferred final workflow; avoid running the test set repeatedly.

## Dataset and methodology

Source: [Cheng brain tumor dataset](https://doi.org/10.6084/m9.figshare.1512427). Cite the original authors and verify dataset terms before redistribution. The two mirrors ([primary](https://github.com/jasonnan2/Deep-Learning-MRI-tumor), [secondary](https://github.com/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning)) contain 3,064 numbered MATLAB files with identical Git blob hashes across mirrors. Original archive identity and actual Kaggle download remain to be confirmed.

| Split | Slices | Patients |
|---|---:|---:|
| Train | 2124 | 163 |
| Validation | 633 | 46 |
| Test | 307 | 24 |

Counts from the earlier Kaggle preprocessing run. Stratified patient split (seed 42), no patient overlaps. Training only: synchronized flips/rotations. Grayscale, foreground percentile normalization (1–99), 256x256 resolution, nearest-neighbor mask resizing. Both architectures use Dice+BCEWithLogits, AdamW, mixed precision on GPU, early stopping and learning-rate scheduling. Attention U-Net uses learned spatial attention gates on skip connections. Evaluation includes Dice, IoU, precision, recall, specificity, patient-macro and slice-macro averages, patient-bootstrap 95% intervals, subgroup metrics and best/worst example plots.

All dataset slices contain tumors; performance on tumor-negative slices, other hospitals and clinical populations is unknown. Predicted image area is not tumor volume or diagnostic confidence. Bootstrap intervals are descriptive, not clinical validation.

## Local dashboard

Extract the Kaggle results ZIP into `neuroseg_results/` at repository root. On PowerShell:

```powershell
pip install -r requirements-v2.txt
$env:NEUROSEG_LOAD_CANDIDATES="1"
python -m uvicorn neuroseg.api:app --port 8000
```

Open http://127.0.0.1:8000. The demo supports de-identified 2D PNG/JPEG/TIFF images, optional matching binary ground-truth masks, architecture selection, side-by-side views, opacity and zoom, mask download, JSON reports and genuine evaluation metrics when the report hash matches the installed checkpoint. Select already-normalized mode only for known preprocessed 8-bit images. No patient data or image content is intentionally logged. Models must come from a trusted training run.

## MLflow and monitoring

The study runner uses a local SQLite backend and local artifacts by default; `--tracking-uri` allows a configured MLflow server. The `NeuroSeg@champion` alias represents the validation-selected model. MLflow artifacts/database are kept in the run directory, not the lightweight submission ZIP. Use `mlflow ui --backend-store-uri sqlite:///ABSOLUTE_PATH_TO/mlflow.db` in the same environment with the artifact directory accessible; registry entries are not portable merely by moving the database. Do not expose an unauthenticated MLflow server publicly.

API endpoints: /health, /api/model/info, /api/model/performance, /api/monitoring, /metrics, /api/predict. Monitoring is per-process latency/error counters, resets on restart, and does not measure model drift or diagnostic quality.

## Testing and deployment

`python -m pytest -q tests` runs model, metric, leakage and API integration tests. Synthetic checkpoints are used only for tests; they are not trained weights. GitHub Actions runs tests and a CPU-container smoke test. See [deployment guide](deploy/README.md) for Docker Compose, optional Prometheus and manual container publishing. Actual cloud hosting requires your account and a genuine checkpoint.

## Status

V2 code is implemented; actual dataset recovery in Kaggle, comparative GPU runs, MLflow registry execution, real metrics, browser testing with trained weights, and live deployment still need verification. Check [Actions](../../actions) for the latest CI outcome. Dependency ranges are provided; CI exports tested versions when it passes. Old root-level V1 scripts and notebook checkpoint artifacts remain for reference only and are not compatible with V2. The historical V1 Dice is not directly comparable. Use the neuroseg-v2 branch until V2 is validated.

Future research: independent external validation, tumor-negative data, calibrated uncertainty, full 3D models, and drift monitoring when representative production data and labels are available.

## Automatic trained-project build (GitHub Actions)

The [Train and package working NeuroSeg workflow](../../actions/workflows/train-and-package.yml) trains both architectures on a standard CPU runner at 128×128, width 16, up to 12 epochs each, selects using validation Dice and tests the selected checkpoint on held-out patients. This is a smaller configuration than the 256×256 Kaggle study, and its results must be identified as such. It verifies actual inference on an existing held-out slice before uploading the `NeuroSeg-trained-windows` artifact. The artifact contains source, both trained weights, aggregate evaluation and a Windows launcher; no raw MRI data or patient-level records are packaged. Download and extract the artifact, then double-click Start-NeuroSeg.bat (requires installed Python and first-run Internet for dependencies). The artifact is available only on successful execution. See the run logs for any blockers. The optional MLflow tracking step is non-blocking; its result must be checked separately.
