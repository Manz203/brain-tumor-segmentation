# V2 execution and verification checklist

- [ ] Execute dataset recovery and verify source hashes/patient split in Kaggle.
- [ ] Train U-Net and Attention U-Net; freeze champion using validation data.
- [ ] Run optional hyperparameter candidates before final test selection.
- [ ] Save actual held-out metrics, plots and model weights.
- [ ] Verify MLflow experiment records and champion alias.
- [ ] Verify latest CI, Docker build and container smoke test.
- [ ] Extract result bundle and verify the browser demo with real trained weights.
- [ ] Deploy privately to an authorized host and verify monitoring.
- [ ] Add real external validation and tumor-negative data if available.
- [ ] Review V2 changes and decide whether to merge into main.
