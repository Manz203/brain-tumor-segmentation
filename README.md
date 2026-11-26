# NeuroSeg V2 — Brain tumor segmentation

Research and educational prototype for binary 2D brain MRI segmentation. Not a clinically validated diagnostic tool.

## Quick start (Kaggle)

Enable GPU and Internet in a private Kaggle notebook. Run the cell in [03_neuroseg_end_to_end.ipynb](Notebooks/03_neuroseg_end_to_end.ipynb). No manually transferred dataset archive is required. The script uses a previously saved processed ZIP or attached original MATLAB files when available, otherwise it attempts a sparse clone of the [dataset mirror](https://github.com/jasonnan2/Deep-Learning-MRI-tumor). The mirror file count was checked; end-to-end download and training still require verification in Kaggle. If access fails, attach the original MATLAB files as an input rather than changing the patient split.

The default notebook run trains up to 15 epochs, selects the best checkpoint on validation Dice, evaluates the fixed baseline once on the test set, and saves best/last checkpoints, metrics, validation history, plots and the pseudonymous split manifest in `/kaggle/working/neuroseg_results`. Use Save & Run All to persist results. GPU training is not yet verified; no V2 scores are claimed. For future exploratory experiments, use `--skip-test` to reserve the test set until model selection. Do not tune the model on test results.

## Dataset and methodology

Source: [Cheng brain tumor dataset](https://doi.org/10.6084/m9.figshare.1512427). Cite the original authors and check dataset terms before redistribution. Original MATLAB files contain T1 contrast-enhanced 2D MRI slices, binary masks and patient identifiers. Prior Kaggle preprocessing execution verified:

| Split | Slices | Patients |
|---|---:|---:|
| Train | 2124 | 163 |
| Validation | 633 | 46 |
| Test | 307 | 24 |

Total: 3064 slices from 233 patients. No patient overlap. Stratified patient-level split with seed 42. The source mirror is checked against these counts and source metadata; identical files and membership have not been independently verified against the original archive. The test set is held out during training and validation-based checkpoint selection. All original data examples contain tumors; generalization to tumor-negative cases is not established.

Preprocessing: grayscale, foreground percentile normalization (1–99), 256x256 training resolution, nearest-neighbor mask resizing, synchronized flips/rotations on training only. U-Net with 32 initial channels, logits, Dice+BCEWithLogits, AdamW, learning-rate scheduling, mixed precision on CUDA, early stopping, sample-weighted metrics. The seed improves reproducibility but does not guarantee bitwise reproducibility. Resume restores model/optimizer/scheduler/scaler, not random generator state.

Evaluation reports Dice, IoU, precision, recall, specificity, patient-macro and slice-macro averages, subgroup metrics and example failure cases.

## Local web demo

After Kaggle training, download `best_unet.pth` into `neuroseg_results/`. In the repository root:

```powershell
pip install -r requirements-v2.txt
python -m uvicorn neuroseg.api:app --port 8000
```

Open http://127.0.0.1:8000. Supports de-identified grayscale MRI PNG/JPEG/TIFF exports; distribution may differ from raw MATLAB scans. Uploaded images are not intentionally persisted. Inference normalization is applied once to uploaded image; already normalized training PNGs may be normalized again and differ from training. No diagnosis, tumor volume or calibrated confidence is provided.

## Tests and container

`python -m pytest -q tests`

`docker build -f Dockerfile.v2 -t neuroseg-v2 .`

`docker run --rm -p 8000:8000 -v "${PWD}/neuroseg_results:/models:ro" neuroseg-v2`

The new code lives in `neuroseg/`. Root-level V1 scripts and old notebook artifacts are retained for reference only; do not use their old checkpoints with the new model. Historical V1 validation Dice of about 0.62 is not comparable with V2 due to the changed split and preprocessing.

## Status and limitations

Implementation ready for execution; dataset recovery, full GPU training, test results, actual deployment and CI execution remain to be verified. Dependencies specify minimums rather than a tested lockfile. Attention U-Net, MLflow integration, external validation, and deployment are future work. No fabricated metrics or model weights are included.
