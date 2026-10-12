# Multimodal 3D AI for Precision Melanoma Intelligence

Multimodal AI for early melanoma detection and risk stratification using
clinical photographs, dermoscopic images, and 3D lesion morphology.

## Objectives
1. Develop a multimodal model integrating clinical, dermoscopic, and 3D lesion information.
2. Add explainable and uncertainty-aware techniques.
3. Validate across diverse Indian skin types and build a prototype decision-support app.

## Status
Project setup. Currently working on Objective 1 (baseline model).

## Setup
python -m venv .venv, activate it, then pip install -r requirements.txt.
Datasets are not stored in this repo; they are downloaded into data/.

Baseline (dermoscopy only, EfficientNet-B0, melanoma vs other, HAM10000): test AUC 0.878, sensitivity 0.699, specificity 0.865.