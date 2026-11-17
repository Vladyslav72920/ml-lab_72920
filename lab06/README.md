# Lab 6 – training pipelines

Setup: `pip install -r requirements.txt` (GPU strongly recommended for Tasks 3-6; peak memory is only measured on CUDA).
Smoke test first: `python studies.py all --quick`.

| Task | Command | Output |
|---|---|---|
| 1 pipeline + MLflow | `python train.py --name main` | `checkpoints/default/`, MLflow run |
| 2 profile | `python profile_epoch.py` | `results/profile_breakdown.{csv,png}` |
| 3 num_workers | `python studies.py workers` | `results/workers_study.{md,png}` |
| 4 batch size | `python studies.py batch` | `results/batch_study.{md,png}` |
| 5 schedulers/optimizers | `python studies.py sched` | `results/sched_optim_study.md`, `sched_study.png`, `optim_study.png` |
| 6 grad accumulation | `python studies.py accum` | `results/accum_*` |
| 7 resume | `python studies.py resume` | `results/resume_curves.png` |
| 8 proposal | edit `PROPOSAL.md` (name + ID), commit | |

View runs: `mlflow ui --backend-store-uri sqlite:///mlflow.db` and take screenshots.
Note: the batch-size study keeps the learning rate fixed, so accuracy differences partly reflect that.
