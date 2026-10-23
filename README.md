# Cognitive Radiology Assistant

A desktop research application for reviewing chest X-rays with a vision-language model. The workspace combines image selection, clinical context, generated report text, and pathology scores in one window.

> **Research use only.** This application is not a medical device and is not a substitute for review by a qualified radiologist. Model scores are estimates, not confirmed diagnoses.

## Current desktop interface

The PySide6 application opens maximized and includes:

- A custom title bar with minimize, maximize/restore, and close controls.
- A connected, animated workflow for image selection, context review, analysis, and output review.
- PNG and JPEG image browsing or drag and drop.
- An editable clinical indication field.
- A report panel with a copy action and a pathology score dot plot on a shared 0 to 100% scale.
- Optional threshold markers and threshold status for model scores.
- A reset action and model status indicator.

## Setup and launch

Use Python 3.10 or newer. From the project directory, install the dependencies into the environment you intend to run the app with:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, allow scripts for the current terminal, then activate in a separate command:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

Alternatively, activate your existing Conda environment and install the requirements there. Launch the GUI with:

```powershell
python run.py --gui
```

You can also start the app directly with `python scripts/gui_app.py` or use `.\run.ps1 --gui` in PowerShell.

## Model weights

The GUI looks for `checkpoints/best_model.pth` at startup and begins loading it in a background thread when present. If it is absent, use **Load model** in the app to choose a checkpoint. The project runner also provides a checkpoint downloader:

```powershell
python run.py --download-model
```

The downloader uses parallel HTTP range requests (24 workers and 8 MiB chunks by default) and resumes completed chunks after an interruption. Tune both for your connection, for example `python run.py --download-model --workers 32 --chunk-mb 16`; higher concurrency may not help if the file host or your internet connection is the limit.

The trained checkpoint is large. The model and tokenizer dependencies may also need to be available locally or downloaded by Transformers the first time they are loaded.

For other runner commands, use:

```powershell
python run.py --help
python run.py --check
python run.py --demo --image "path\to\chest-xray.jpg"
python run.py --test
```

## Important: bundled reference cases

Three bundled images are provided in [`demo_images/`](demo_images/):

- `sample_normal_adult_pa.png`
- `sample_pediatric.jpg`
- `sample_pleural_mass.jpg`

When one of these images (or a pixel-identical copy) is selected and **Run analysis** is pressed, the app displays its bundled reference report and scores. It does **not** run model inference for those images. For other images, the app runs inference when model weights are loaded. Treat the sample outputs as fixed examples, not model predictions or clinical advice.

## Model overview

The model code combines:

1. **PRO-FA** - hierarchical visual features from a Vision Transformer.
2. **MIX-MLP** - scores for 14 chest X-ray findings.
3. **RCTA** - fusion of image, clinical-context, and finding features.
4. **Report generation and threshold-based grounding** - generated report text is refined using pathology scores and configurable thresholds.

These components describe the implementation; they do not establish clinical performance or safety. Review the source under [`models/`](models/) and [`scripts/`](scripts/) for implementation details.

## Repository map

| Path | Purpose |
| --- | --- |
| `scripts/gui_app.py` | PySide6 desktop interface and inference flow |
| `models/` | Model architecture and dataset transforms |
| `training/` | Training pipeline |
| `evaluation/` | Evaluation scripts |
| `demo_images/` | Bundled example X-rays |
| `outputs/` | UI prototypes and audit screenshots |
| `tests/` | Model tests and UI audit harness |
| `run.py` | Project command-line runner |
| `requirements.txt` | Python dependencies |
