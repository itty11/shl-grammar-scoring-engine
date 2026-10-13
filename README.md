# SHL Grammar Scoring Engine — Hiring Assessment 2026

## Overview

This project implements a machine learning pipeline for the SHL Hiring Assessment 2026 Grammar Scoring Engine challenge.

The objective is to predict a grammar score from short spoken audio recordings. The workflow explores audio transcription, text-based features, signal-derived features, regression models, and model evaluation.

## Project Workflow

The project includes the following components:

- **Audio and dataset inspection:** Scripts for exploring dataset structure and audio characteristics.
- **Transcription:** Scripts for generating transcripts from training and test audio.
- **Text-based modelling:** A baseline model using transcript-derived features.
- **Multimodal modelling:** Experiments combining text and audio signal features.
- **Model evaluation:** Cross-validation and validation prediction analysis.
- **Hyperparameter tuning:** Experiments to select model settings.
- **Prediction generation:** Generation of predictions for the assessment test set.

## Repository Structure

- `SHL_Grammar_Scoring_Engine.ipynb` — Main experimentation notebook.
- `baseline_text_model.py` — Text-based baseline.
- `baseline_multimodal.py` — Multimodal modelling experiment.
- `cross_validate_model.py` — Cross-validation workflow.
- `tune_ridge.py` — Ridge regression tuning.
- `transcribe_train.py` — Training audio transcription.
- `transcribe_test.py` — Test audio transcription.
- `inspect_dataset.py` — Dataset inspection.
- `inspect_audio.py` — Audio inspection.
- `inspect_signal.py` — Signal feature inspection.
- `inspect_test_signal.py` — Test signal inspection.
- `inspect_transcripts.py` — Transcript inspection.
- `check_mapping.py` and `check_sample.py` — Data mapping and sample checks.
- `cross_validation_results.csv` — Cross-validation results.
- `ridge_tuning_results.csv` — Hyperparameter tuning results.
- `submission_candidate.csv` — Candidate prediction output.

## Requirements

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Use a compatible Python environment and ensure that any additional transcription or audio-processing dependencies required by the scripts are installed.

## Dataset

The original audio recordings and assessment data are not included in this public repository. Use the authorized assessment or competition source to obtain the required files, subject to its access and usage rules.

Place the permitted dataset files in the expected local directory structure before running the relevant scripts.

## Running the Project

1. Install the required dependencies.
2. Obtain the dataset through the authorized source.
3. Run the appropriate transcription and feature-extraction steps.
4. Execute the modelling and validation scripts.
5. Review the evaluation results.
6. Generate and verify the final prediction file against the official submission format.

Check the individual scripts and notebook for implementation details and execution order.

## Evaluation

The project includes cross-validation, validation prediction analysis, and model-tuning experiments.

Refer to the corresponding result files and notebook for the actual metrics and comparisons. Final performance should be reported using verified evaluation results.

## Final Output

`submission_candidate.csv` contains the candidate predictions prepared for submission. Confirm that it matches the official required schema and the exact output intended for the assessment before submitting it.

## Notes

This repository documents the implementation and experiments for the SHL Grammar Scoring Engine hiring assessment. Dataset redistribution and publication of derived data must comply with the applicable assessment and competition rules.