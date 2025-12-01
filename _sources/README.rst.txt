Welcome to the CuBATS Documentation!
====================================

CuBATS (Computer vision-Based Antigen Target Selector) is an open-source computer vision pipeline for patient-specific tumor-associated antigen (TAA)
selection to support multi-target CAR T-cell strategies. It analyzes a variable number of
immunohistochemically (IHC) stained whole-slide images (WSIs), alongside a single hematoxylin and
eosin (HE) stained WSI for tumor segmentation. CuBATS systematically quantifies antigen expression and
identifies optimal mono-, dual-, or triplet combinations for multi-targeted CAR T-cell therapy design —
maximizing spatial tumor coverage while minimizing TAA overlap. CuBATS can be applied to any solid tumor
type, given an appropriate tumor segmentation model.

CuBATS integrates WSI registration, tumor segmentation, color deconvolution, quantification, and
combinatorial analysis into a unified, streamlined framework. It enables patient-specific, reproducible,
and scalable TAA selection addressing the challenges of spatial tumor heterogeneity and antigen escape.

Pipeline Overview
-----------------
CuBATS includes the following steps:

1. **WSI Registration:** Registration and alignment of tissue across all WSIs using VALIS
   (`VALIS <https://github.com/MathOnco/valis>`_).
2. **Tumor Segmentation:** Configurable tumor segmentation based on the HE WSI. CuBATS accepts
   segmentation models in ``ONNX`` format.
3. **Color Deconvolution:** Separation of antigen-specific DAB stain from counterstains.
4. **Quantification of Staining Intensities:** Classification of tissue regions into high-, medium-,
   low-positive and negative intensity categories.
5. **Combinatorial Analysis:** Evaluation of spatial TAA co-expression to identify optimal mono-, dual-,
   and triplet TAA combinations that maximize tumor coverage while minimizing TAA overlap.

Installation & Examples
-----------------------
For installation instructions refer to the :doc:`installation` section.

For runnable walkthroughs and code examples see the :doc:`Examples <examples>` section.

Contributing
------------
If you want to contribute to the project please check out our `Contributing Guidelines <https://github.com/hnu-digihealth/CuBATS/blob/main/CONTRIBUTING.md>`_ on GitHub.


License
-------
`MIT <LICENSE.txt>`_ © 2025 Moritz Dinser, Daniel Hieber
