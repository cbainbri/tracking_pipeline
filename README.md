# Tracking Pipeline

Worm tracking tools for both food encounter (food) and optogenetics (opto) experiments.

## Workflow

### 1. Preview and tune parameters — `track_preview.py`

**Start here.** Run `track_preview.py` on a single representative directory before committing to a full batch. It lets you quickly see how well the tracker performs with the current parameters (detection thresholds, minimum track length, pixel scale, etc.) so you can iterate without waiting for a full batch run.

Once you are satisfied with the results, copy the tuned parameter values into `batch_tracking_food.py` before running the full dataset.

```
python track_preview.py
```

### 2. Batch tracking

After parameters are tuned in `track_preview.py`, run the appropriate batch tracker for your experiment type:

| Script | Use for |
|---|---|
| `batch_tracking_food.py` | Food encounter experiments (standard) |
| `batch_tracking_gpu_accel_food.py` | Food encounter experiments (GPU-accelerated) |
| `batch_tracking_opto.py` | Optogenetics experiments |

```
python batch_tracking_food.py
# or
python batch_tracking_opto.py
```

### 3. Edit tracks — post-tracking QC

Use the track editors to manually correct tracking errors after the batch run:

| Script | Use for |
|---|---|
| `track_editor_food.py` | Food encounter tracks |
| `track_editor_opto.py` | Optogenetics tracks |

## Output

Batch trackers produce wide-format CSVs (one row per frame, one column group per worm). These are the input files for the downstream analysis repos ([food_encounter_analysis](https://github.com/cbainbri/food_encounter_analysis), [opto_analysis](https://github.com/cbainbri/opto_analysis)).

## Installation

```
pip install -r requirements.txt
```

> Linux users: Tkinter is not pip-installable. Install via `sudo apt-get install python3-tk`.
> GPU users: the default torch install is CPU-only. See [PyTorch installation](https://pytorch.org/get-started/locally/) for GPU builds.
