# Smart device usage and student mental health risk

Code for

**Smart Device Usage and Student Mental Health Risk Assessment: A Probabilistic Prediction Framework Integrating Dynamic Relationship Modeling and Uncertainty Calibration**

Zeng Wenquan · https://github.com/chaosbull

The model (DRG-MDN-U) predicts a mental-health score from daily device use, sleep, study, activity, and a few demographic fields. It also returns a standard deviation, so a 90% interval is just the prediction plus or minus 1.645 sigma. A single scale, fit on the validation split, is applied afterwards. On this run that scale is 1.18.

## Layout

```
data/            the 5,000-row table
src/             training, the model, and the plotting code
checkpoints/     drg_mdn_best.pt and the earlier DBIAN weights
```

Nothing from the reported run is checked in except those two weight files. `python src/run.py` creates `results/` and writes the figures and tables there.

## Test set (seed 42)

Coverage for DRG-MDN-U is after the 1.18 scale. Before that scale, 90% coverage on the same predictions is 85.5%, and ECE is 0.024.

| Model | RMSE | MAE | R² | NLL | 90% coverage |
| --- | --- | --- | --- | --- | --- |
| DRG-MDN-U | 0.373 | 0.264 | 0.921 | 0.445 | 0.893 |
| DBIAN | 0.416 | 0.292 | 0.901 |  |  |
| Gradient boosting | 0.485 | 0.389 | 0.865 | 0.696 | 0.912 |
| Linear regression | 0.693 | 0.547 | 0.726 | 1.052 | 0.896 |

DBIAN is not retrained here. `src/run.py` loads `checkpoints/dbian_best.pt` and scores it on the same test rows. If that file is missing, the comparison table just omits it.

The 1.18 scale was chosen so validation coverage sits near 90%. It does not improve ECE. Raw ECE is 0.024; after scaling it is 0.040.

## Setup

Python 3.10 or newer. CUDA is used if it is available. The saved run trained for 286 epochs.

```bash
pip install -r requirements.txt
python src/run.py
```

Run it from the repository root. The script puts `src/` on the path itself.

Split is 70% / 15% / 15%, `seed = 42`. Sleep hours and daily usage are Box-Cox transformed on the full table before the split, then all continuous columns are z-scored the same way.

## What the network actually does

Four categorical columns (country, academic level, platform, purpose) go through 8-d embeddings. Eight scaled base columns and four products built from them (`Digital_Saturation`, `Restorative_Ratio`, `Behavioral_Balance`, `Stress_Usage_Coupling`) make up the rest. The vector is 44-d. Each dimension is a graph node with its own type embedding.

Edges are a 4-head attention, gated by the stress value, then cut down to the 6 strongest neighbors. Two residual graph-conv layers follow. A point head predicts the score; the three Gaussian means are that prediction plus a residual, and the reported point forecast is the mixture mean, not the point head by itself. While training, the per-row sigma scale is not applied. It is applied at evaluation, with a floor of 1.10, and the validation scale (1.18) is multiplied on top.

About 37k parameters. Loss is the mixture NLL, Huber on the mixture mean, a smaller Huber on the point head, a little entropy on the mixture weights, and a VAT term after epoch 20. Early stopping watches validation RMSE. From epoch 80 the weights are averaged; the average is kept only if its validation RMSE is better.

## Data

`data/student_social_media_mental_health.csv`, 5,000 rows. Columns:

`Age`, `Gender`, `Country`, `Academic_Level`, `Most_Used_Platform`, `Purpose_Of_Use`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Study_Hours`, `Physical_Activity_Hours`, `Sleep_Hours_Per_Night`, `Stress_Level`, `Mental_Health_Score`.

Stress is mapped Low / Medium / High / Very High to 0.25 / 0.50 / 0.75 / 1.00. Gender is 0/1.

## Citation

```bibtex
@misc{zeng2026deviceMH,
  author = {Zeng, Wenquan},
  title  = {Smart Device Usage and Student Mental Health Risk Assessment: A Probabilistic Prediction Framework Integrating Dynamic Relationship Modeling and Uncertainty Calibration},
  year   = {2026},
  url    = {https://github.com/chaosbull/smart-device-mh-risk}
}
```

## License

Code and weights are under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). See `LICENSE`.
