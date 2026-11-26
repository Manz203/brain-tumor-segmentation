# Deploy the research demo

## Local or cloud VM (Docker Compose)

1. Install Docker and the Compose plugin. On a VM, allow SSH only; do not expose unauthenticated research endpoints to the internet.
2. Clone the neuroseg-v2 branch. Extract the actual Kaggle submission ZIP into neuroseg_results/ in the repository root. Ensure the read-only volume is readable by UID 10001.
3. Run `docker compose up --build -d`. For optional local Prometheus use `docker compose --profile monitoring up --build -d`.
4. Verify `curl http://127.0.0.1:8000/health` returns {"ready":true}. False means no model loaded. Visit http://127.0.0.1:8000. Prometheus is at http://127.0.0.1:9090 if enabled.
5. For remote access use SSH forwarding, e.g. `ssh -L 8000:127.0.0.1:8000 USER@VM_HOST`. For broader access, configure HTTPS, authentication, upload/rate limits, and a data-handling policy before exposing it. Do not use patient records.

Metrics: neuroseg_predictions_total, neuroseg_prediction_errors_total and neuroseg_prediction_duration_seconds. These are operational metrics only. Verify them with an actual prediction. Counters reset when the server restarts.

## Publish a container

The manual GitHub Actions workflow Publish NeuroSeg container can push a CPU image tagged with the commit SHA to GHCR using the repository GITHUB_TOKEN. Run it only after CI passes and confirm package permissions. The image does not contain model weights; mount the trusted checkpoint/results directory at /models. Publishing a container is not the same as deploying a live service.

## Verification checklist

- Check successful CI and Docker build.
- Install trusted weights and verify readiness and model information.
- Predict on a de-identified supported image; confirm original, mask and overlay.
- Validate ground-truth comparison and report provenance.
- Verify bad uploads produce clear 4xx responses and monitoring increments.
- Confirm evaluation metrics match the actual installed checkpoint.
- Confirm the deployment is private or protected, and record the deployed commit/model hash.

No cloud account, VM, public URL or credentials are provisioned by this repository.
