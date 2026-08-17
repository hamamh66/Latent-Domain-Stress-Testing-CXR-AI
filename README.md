# [Latent-Domain Stress Testing of CXR AI](https://github.com/hamamh66/Latent-Domain-Stress-Testing-CXR-AI)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamamh66/Latent-Domain-Stress-Testing-CXR-AI/blob/main/Latent_Acquisition_Domain_Stress_Testing_CXR_AI.ipynb)

Reproducible, storage-safe experiments for auditing shortcut learning in pneumonia and tuberculosis chest X-ray triage through duplicate control, latent acquisition-domain reconstruction, held-out-domain testing, source balancing, calibration, conformal prediction, and selective referral.

## Overview

This repository accompanies the manuscript **“Latent Acquisition-Domain Stress Testing of Artificial Intelligence for Pneumonia and Tuberculosis Triage on Chest Radiographs,”** prepared for *Thoracic Radiology*.

The notebook investigates whether strong internal chest X-ray classification performance persists when acquisition-related technical domains are withheld. It combines data-integrity auditing, latent-domain reconstruction, source-neutralization ablations, robust training, calibration, conformal prediction, and selective referral in one executable workflow.

> **Evidentiary boundary:** The completed study reconstructed two latent technical domains because verified upstream provenance was unavailable. These domains must not be interpreted as identified hospitals, scanners, or repositories. The study is a domain-shift stress test, not independent external clinical validation.

## Main notebook

[`Latent_Acquisition_Domain_Stress_Testing_CXR_AI.ipynb`](Latent_Acquisition_Domain_Stress_Testing_CXR_AI.ipynb) is the complete Colab-ready workflow. It performs:

1. dataset retrieval and configuration;
2. SHA-256 and perceptual duplicate auditing;
3. exclusion of contradictory-label identities;
4. deterministic provenance matching when a manifest or upstream roots are supplied;
5. latent acquisition-domain reconstruction for unmatched images;
6. source–label association and domain-predictability analysis;
7. duplicate-group-aware internal splitting;
8. input-intervention and source-neutralization ablations;
9. frozen MobileNetV3 embedding extraction;
10. ERM, source-balanced, GroupDRO, and CORAL-style classifiers;
11. internal and eligible leave-one-domain-out evaluation;
12. temperature scaling, class-conditional conformal prediction, and selective referral;
13. stratified bootstrap analysis of the primary outcome;
14. optional external evaluation when an independent cohort is explicitly configured;
15. manuscript-ready PNG figures, CSV tables, JSON evidence, and a compact results archive.

## Principal completed-run results

| Analysis | Result |
|---|---:|
| Audited files | 25,553 |
| Unique SHA-256 identities | 19,962 |
| Retained duplicate-controlled images | 14,281 |
| Latent domains | 2 |
| Domain–label Cramér’s V | 0.7284 |
| Native-metadata domain prediction, macro-F1 | 0.9986 |
| Internal ERM macro-F1 | 0.9897 |
| Held-out L1 ERM macro-F1 | 0.8806 |
| Held-out L2 ERM macro-F1 | 0.8037 |
| Aggregated held-out minus internal macro-F1 | −0.0907 |
| 95% bootstrap interval | −0.0975 to −0.0836 |
| Source-balanced macro-F1, L1 / L2 | 0.9057 / 0.8310 |

Under ERM, expected calibration error increased from 0.0066 internally to 0.0462–0.0578 under domain shift. Coverage from class-conditional conformal prediction decreased from 0.9071 internally to 0.7856 and 0.6972 on the two held-out domains, despite a nominal 0.90 target.

## Important robustness limitation

Only two latent domains were reconstructed. Consequently, withholding one domain leaves only one training domain. In those folds:

- GroupDRO has a single optimization group;
- the CORAL between-domain alignment penalty is zero; and
- their held-out predictions collapse to ERM by construction.

Their equality with ERM is therefore a design consequence and must not be presented as evidence that GroupDRO or CORAL is generally ineffective. At least three verified, label-complete development domains are needed for an informative held-out comparison of multi-domain robustness objectives.

## Dataset

The default configuration retrieves the public [Chest X-Ray Dataset](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset) through `kagglehub`. Expected labels are:

- `normal`
- `pneumonia`
- `tuberculosis`

Source images are not included in this repository. Their use remains subject to the dataset provider’s terms and documentation.

## Running in Google Colab

1. Open the notebook using the Colab badge above.
2. Select **Runtime → Change runtime type → T4 GPU** or another available GPU.
3. Review the `CONFIG` dictionary in Section 2.
4. Select **Runtime → Run all**.
5. Download the generated compact ZIP if Google Drive is unavailable or full.

The dataset is downloaded automatically when it is not already present. A GPU is strongly recommended for embedding extraction, although the downstream classifiers are lightweight.

## Storage-safe behavior

The workflow was designed for a nearly full Google Drive:

- datasets, embeddings, checkpoints, split assignments, and case-level predictions remain in temporary Colab storage;
- only one compact ZIP is offered to Drive;
- the default Drive-export cap is 25 MB;
- if that limit is exceeded, PNG figures are omitted from the Drive copy while numerical evidence is retained;
- if Drive rejects the copy, an incomplete file is removed and browser download is triggered; and
- a full runtime archive is disabled by default.

The cap can be changed deliberately through `CONFIG["max_drive_export_mb"]`.

## Optional provenance and external-validation inputs

The following configuration fields may be supplied before execution:

```python
CONFIG["provenance_manifest_csv"] = "/path/to/provenance_manifest.csv"
CONFIG["upstream_roots"] = {
    "verified_source_name": "/path/to/upstream/source"
}
CONFIG["external_roots"] = {
    "external_cohort_name": "/path/to/independent/cohort"
}
```

External results are generated only when an independent cohort is explicitly configured. Blank radiologist-review fields are never interpreted as completed clinical review.

## Generated outputs

The runtime output directory contains:

```text
Thoracic_Radiology_Source_Aware_CXR/<timestamp>/
├── Figures/                  # Manuscript-ready PNG figures
├── Tables/                   # Main and diagnostic CSV tables
├── Raw/                      # Configuration and audit guardrails
├── Models/                   # Temporary model artifacts
├── Outputs_Summary.txt       # Human-readable run summary
└── experiment_results.json   # Principal machine-readable evidence
```

The compact archive includes the four main manuscript tables, diagnostic tables, figures when permitted by the cap, integrity metadata, the primary bootstrap outcome, configuration, and the run summary.

## Reproducibility settings

The default experiment uses:

- image size: 192 × 192 pixels;
- encoder: ImageNet-pretrained `mobilenetv3_small_100`;
- frozen image embeddings;
- three training seeds: 20260816, 20260817, and 20260818;
- 45 maximum epochs with early stopping;
- 1,000 bootstrap replicates;
- conformal miscoverage level: 0.10; and
- selective-acceptance threshold fitted at the 75th percentile of calibration entropy.

The complete executed configuration is exported as `Raw/run_config.json`. Minor numerical differences may arise across hardware, CUDA, PyTorch, and library versions.

## Software

The notebook uses Python, PyTorch, torchvision, timm, scikit-learn, SciPy, Pillow, ImageHash, NumPy, pandas, Matplotlib, seaborn, joblib, and kagglehub. Missing notebook-specific packages are installed automatically in Colab.

## Responsible interpretation

The workflow is intended for retrospective research and methodological auditing. It does not provide autonomous diagnosis, establish prospective clinical utility, or replace local validation. Public pneumonia and tuberculosis labels may not be equivalent to adjudicated radiological or microbiological reference standards.

## Citation

Until final publication metadata are available, the repository may be cited as:

```bibtex
@unpublished{Mallek2026LatentDomainCXR,
  author  = {Mallek, Fatma and Zayoud, Rahma and Hamam, Habib},
  title   = {Latent Acquisition-Domain Stress Testing of Artificial Intelligence
             for Pneumonia and Tuberculosis Triage on Chest Radiographs},
  year    = {2026},
  note    = {Manuscript prepared for Thoracic Radiology},
  url     = {https://github.com/hamamh66/Latent-Domain-Stress-Testing-CXR-AI}
}
```

The citation should be replaced with the article DOI and final bibliographic information after publication.

## Authors

- Fatma Mallek
- Rahma Zayoud
- Habib Hamam

## License and data terms

A software license should be selected before public release. Dataset files are not redistributed, and the upstream dataset’s terms remain applicable independently of the eventual code license.
