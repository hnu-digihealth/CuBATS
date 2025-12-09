# **C**omp**u**ter vision-**B**ased **A**ntigen **T**arget **S**elector <img src="https://raw.githubusercontent.com/hnu-digihealth/CuBATS/main/docs/images/CuBATS_logo.png" alt="CuBATS" title="CuBATS Logo" width="150" height="150" />


[![codecov](https://codecov.io/gh/hnu-digihealth/CuBATS/graph/badge.svg?token=5R3EBA6JS9)](https://codecov.io/gh/hnu-digihealth/CuBATS)
[![CI](https://github.com/hnu-digihealth/CuBATS/actions/workflows/test.yaml/badge.svg)](https://github.com/hnu-digihealth/CuBATS/actions/workflows/test.yaml)

**CuBATS** (Computer vision-Based Antigen Target Selector) is an open-source computer vision pipeline for patient-specific tumor-associated antigen (TAA) selection to support multi-target CAR T-cell strategies. It analyzes a variable number of **immunohistochemically (IHC) stained whole-slide images (WSIs)**, alongside a single **hematoxylin and eosin (HE) stained WSI** for tumor segmentation. CuBATS systematically quantifies antigen expression and identifies optimal mono-, dual-, or triplet combinations for **multi-targeted CAR T-cell therapy design**&mdash;maximizing spatial tumor coverage while minimizing TAA overlap. CuBATS can be applied to any solid tumor type, given an appropriate tumor segmentation model.

CuBATS integrates WSI registration, tumor segmentation, color deconvolution, quantification, and combinatorial analysis into a unified, streamlined framework. It enables **patient-specific, reproducible, and scalable** TAA selection addressing the challenges of **spatial tumor heterogeneity** and **antigen escape** that currently hinder translation of CAR T-cell therapy to solid tumors.

## Pipeline Overview
CuBATS includes the following steps:
1. **WSI Registration:** Registration and alignment of tissue across all WSIs using [VALIS](https://github.com/MathOnco/valis).
2. **Tumor Segmentation:** Configurable tumor segmentation based on the HE WSI. CuBATS accepts segmentation models in **ONNX** format, enabling tumor-type independent application, if an appropriate model is provided.
3. **Color Deconvolution:** Separation of antigen-specific DAB stain from counterstains.
4. **Quantification of Staining Intensities:** Classification of tissue regions into high-, medium-, low-positive and negative intensity categories.
5. **Combinatorial Analysis:** Evaluation of spatial TAA co-expression to identify optimal mono-, dual-, and triplet TAA combinations that maximize tumor coverage while minimizing TAA overlap.

For full details and installation instructions, see the [Documentation](https://hnu-digihealth.github.io/CuBATS/).

## Contribute to the Project
If you want to contribute to the project please read our [Contributing Guideline](https://github.com/hnu-digihealth/CuBATS/blob/main/CONTRIBUTING.md).


## License
[MIT](https://github.com/hnu-digihealth/CuBATS/blob/main/LICENSE.txt) © 2025 Moritz Dinser, Daniel Hieber
