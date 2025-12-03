.. _Examples:

Examples
********

CuBATS processes a single HE-stained WSI, used for tumor segmentation, along with a variable number of IHC-stained WSIs, each stained for a tumor-associated antigen (TAA).
All downstream analyses—including registration, quantification, and combinatorial co-expression analysis—are performed on these WSIs.

The pipeline leverages **multiprocessing** to speed up computation, and supports both **CPU** and **GPU** execution.
Hardware availability is automatically detected, with seamless fallback to **CPU** if no compatible **GPU** is found.
**GPU**-based processing uses `CuPy <https://cupy.dev>`_, whereas **CPU**-based processing relies on standard `NumPy <https://numpy.org>`_ operations.
In both cases, computations build upon tile-based parallelization and vectorization, providing substantial performance gains for computationally intensive steps such as quantification and multi-antigen analysis.

The following examples provide a complete walkthrough of CuBATS and serve as a guide to get
started. They cover the main steps of the pipeline, starting with *image registration* and *tumor segmentation*, followed by *quantification* and *combinatorial analysis* of antigen expressions.

An executable version of the these examples can be found as a Jupyter notebook in the `CuBATS/examples` directory.

.. note::
    In this documentation, the terms *slide* and *whole-slide image (WSI)* are used interchangeably.
    For clarity, “WSI” is preferred when referring to the digital image data processed by CuBATS.

.. important::
    Depending on the size of your data, ensure your system has sufficient disc space and RAM available for processing.  We recommend a minimum of 32 GB of RAM, or the use of a dedicated server for larger datasets.

.. _Initialize SlideCollection:

Initialize SlideCollection
==========================

CuBATS organizes and processes WSIs in the :class:`SlideCollection`.
Once initialized, the collection provides built-in methods for *registration*, *tumor segmentation*, *quantification*, and *combinatorial antigen analysis*.

`Example 1`_ demonstrates the initialization of a **SlideCollection**. A **SlideCollection** can be initialized with the following arguments: :code:`collection_name` (e.g., tumor set or patient ID), a source directory (:code:`src_dir`) containing the WSIs, a destination directory (:code:`dest_dir`) for results, an optional reference WSI (:code:`ref_slide`), and an optional path to antigen threshold profiles (:code:`path_antigen_profiles`).
If :code:`ref_slide` is omitted, CuBATS automatically selects an HE WSI based on filenames. If no :code:`path_antigen_profiles` is provided, default thresholds are used.

.. note::
    If the specified :code:`dest_dir` already contains previous processing results, these are automatically reloaded when initializing the collection, allowing to resume analysis without reprocessing completed steps.
    Reprocessing will overwrite existing results, so back up any important data before re-running analyses.

.. _Example 1:

Example 1: Initialize SlideCollection
--------------------------------------
.. code-block:: python

    from cubats.slide_collection.slide_collection import SlideCollection

    my_collection = SlideCollection(
        collection_name  "Tumor_Set_01",
        src_dir = "/path/to/wsi_files",
        dest_dir = "/path/to/output_dir",
        ref_slide = "/path/to/reference_wsi.tiff",
        path_antigen_profiles = "/path/to/threshold_profiles.json"
    )


.. _Image Registration:

Image Registration
==================

.. important::
    Image registration and alignment is performed using the `VALIS <https://github.com/MathOnco/valis>`_ framework. Please ensure all dependencies are correctly installed. For details, refer to the CuBATS :ref:`Installation` section or the `VALIS documentation <https://valis.readthedocs.io/en/latest/index.html>`_.

    CuBATS provides a wrapper class that predefines registration parameters for convenience. For more customized registration or if the registration results are unsatisfactory, check out the `VALIS documentation <https://valis.readthedocs.io/en/latest/index.html>`_ for parameter adjustments.

This example demonstrates registering a collection of WSIs and aligning them.
Registration can be performed towards a selected reference WSI or automatically towards a WSI chosen by VALIS. While reference-based registration may be beneficial in some cases, we have observed that automatic registration often produced more accurate results for larger datasets during development.

By default, registration includes a *rigid registration* followed by a *non-rigid registration*. Optionally, high-resolution *micro-registration* can be enabled via :code:`microregistration = True`, which is recommended for large high-resolution WSIs.

- `Example 2`_ shows registration *with a reference WSI*.
- `Example 3`_ shows registration *without a reference WSI*.

WSIs must be located in :code:`SlideCollection.src_dir`. Registered WSIs will be saved to :code:`SlideCollection.registration_dir`. The parameter :code:`max_non_rigid_registration_dim_px` defaults to 2000 for high-resolution registration but can be adjusted, as in `Example 2`_. Cropping can be specified via :code:`crop` with options `"overlap"`, `"reference"`, or `None`. Micro-registration is be enabled via :code:`microregistration=True`.

.. _Example 2:

Example 2: Image Registration With Reference WSI
--------------------------------------------------

.. code-block:: python

    my_collection.register_slides(
        reference_slide = "path/to/reference_wsi.tiff",
        microregistration = True,
        crop = "reference"
    )

.. _Example 3:

Example 3: Image Registration Without Reference
-----------------------------------------------

.. code-block:: python

     my_collection.register_slides(
        microregistration = True,
        max_non_rigid_registration_dim_px = 1800,
        crop = "overlap"
    )

.. note::
        Additional registration outputs and intermediate files are stored inside the `intermediate_registration_results` directory. For more details on subdirectory contents or advanced registration options, see the `VALIS documentation <https://valis.readthedocs.io/en/latest/index.html>`_.


.. _Image Segmentation:

Tumor Segmentation
==================

.. important::
    The accuracy of tumor segmentation depends on the chosen model and its training. Segmentation parameters such as :code:`tile_size`, :code:`normalization`, or :code:`inversion` may require adjustment. If results are unsatisfactory, modify the parameters or verify the model quality.

`Example 4`_ demonstrates running tumor segmentation using :code:`my_collection.tumor_segmentation`. The function applies a segmentation model to the HE-stained WSI and produces a binary tumor mask (.TIFF). An HE-stained WSI is required for this step, as it better captures morphological tissue structures and tumor boundaries than antigen-specific stains such as IHC. The input can be a single WSI file or a directory containing multiple HE-stained WSIs.

The segmentation model must be provided as an `.ONNX` file via :code:`model_path`. The :code:`tile_size` parameter is specified as a tuple (e.g., (1024, 1024)). Optional parameters include :code:`output_path` (default is :code:`SlideCollection.registration_dir`), :code:`normalization` (Reinhard normalization), :code:`inversion` (invert mask), and :code:`plot_results` (creates a thumbnail overlay of the tumor mask on the WSI).

The function automatically resizes input tiles to match the model's expected input size and scales the output mask back to the original tile size, after segmentation.

.. _Example 4:

Example 4: Tumor Segmentation
------------------------------

.. code-block:: python

    my_collection.tumor_segmentation(
        model_path = "path/to/model.onnx",
        reference_slide = "path/to/he_wsi.tiff",
        tile_size = (512, 512),
        output_path = None,
        normalization = False,
        inversion = False,
        plot_results = True
    )

.. _Quantify SlideCollection:

WSI Quantification
==================

Quantification in CuBATS is performed on the previously registered IHC-stained WSIs using their extracted antigen-specific DAB stain channel. The DAB channel is separated from the hematoxylin channel using color deconvolution [1]_, allowing accurate measurement of antigen staining.
Each WSI is divided into non-overlapping tiles of 1024×1024 pixels, which are quantified individually in a pixel-wise manner. For each tile, the staining intensities are measured across tumor regions, and the results are ultimately aggregated for the entire WSI.
This step implements a variation of the IHC-Profiler algorithm [2]_, producing output metrics such as *tumor coverage*, stratification of coverage into *high, medium-, and low-positive expression*, *negative tissue*, *background*, *H-score*, and an additional *IHC-Profiler score*.

CuBATS allows quantification of either all IHC WSIs in the **SlideCollection** or a single slide individually.
Results are automatically stored as :code:`.CSV` and :code:`.PICKLE` files in the specified :code:`data_dir`.

Quantification Modes
--------------------

Quantification can be run in two mask application modes:

- :code:`"tile-level"` (default): Applies the tumor mask coarsely — tiles overlapping the mask are fully included.
  Recommended when registration accuracy is moderate.
- :code:`"pixel-level"`: Applies the tumor mask at pixel precision — only masked pixels are included.
  Offers higher accuracy but is more sensitive to registration noise.

CuBATS also supports *antigen-specific threshold profiles*, allowing for fine-tuned quantification across different antibodies or staining intensities.
Thresholds can be supplied as a :code:`.JSON` or :code:`.CSV` file via the :code:`threshold_profile_path` parameter when initializing the :class:`SlideCollection`.
If omitted, default thresholds are used.

CuBATS also offers post-quantification reconstruction of tiles into a DAB WSI. In order to do this, saving of tile images must be enabled via :code:`save_imgs=True` in the quantification functions.

Quantify All IHC WSIs in SlideCollection
-----------------------------------------

`Example 5`_ demonstrates how to quantify all IHC WSIs within a predefined :class:`SlideCollection`.
All WSIs (except the reference and mask slides) are processed sequentially, and results are saved to the output directory. This example applies `"tile-level"` masking without saving tile images.

.. _Example 5:

Example 5: Quantify All WSIs
----------------------------

.. code-block:: python

    from cubats.slide_collection.slide_collection import SlideCollection

    # Quantify all slides in the collection
    my_collection.quantify_all_slides(
        save_imgs = False,
        masking_mode = "tile-level"
    )

After execution, the results are available in :code:`my_collection.quantification_results`
and stored as :code:`tile-level_quantification_results.csv` inside the :code:`data_dir`.

Quantify a Single Slide
-----------------------

`Example 6`_ shows how to quantify a single slide within a predefined :class:`SlideCollection`.
This is useful for re-quantifying a specific slide with different parameters. This example applies `"pixel-level"` masking and saves tile images for potential reconstruction.

.. _Example 6:

Example 6: Quantify a Single Slide
------------------------------------

.. code-block:: python

    from cubats.slide_collection.slide_collection import SlideCollection

    # Quantify a single slide by name
    my_collection.quantify_single_slide(
        slide_name = "Slide_01",
        save_img = True,
        masking_mode = "pixel-level"
    )

.. note::
    Quantification overwrites existing results in :code:`data_dir`.
    Back up previous data before re-running quantification on the same collection.



.. _Combinatorial Analysis of Antigen Expressions:

Combinatorial Analysis of Antigen Expressions
=============================================

CuBATS enables spatial co-expression analysis of TAAs by performing pixel-wise comparisons across the previously quantified IHC-stained WSIs.
For each antigen pair or triplet, CuBATS computes combined tumor coverage, identifying regions of *overlapping expression* and *complementary expression* between markers as well as stratification into *high-, medium-, and low-positive* categories.
This analysis builds upon the results generated during quantification, using the same antigen-specific intensity thresholds.

Results are stored in :code:`.CSV` and :code:`.PICKLE` format within the :code:`data_dir`.
Optionally, a visualization of the combinatorial analysis can be saved for reconstruction into a WSIs. This can be done by enabling :code:`save_imgs=True`.

Furthermore the selection of tile-level or pixel-level masking can be specified via the :code:`masking_mode` parameter. This parameter should be set consistent with the selection used during quantification.

Pairwise-Antigen Combinations
-------------------------------

`Example 7`_ demonstrates how to compute all possible antigen pairs within a predefined :class:`SlideCollection`.
Each pair of quantified slides is compared tile by tile, with results aggregated to a WSI-level. The example applies `"pixel-level"` masking with saving tile images.

.. _Example 7:

Example 7: Pairwise-Antigen Co-Expression Analysis
---------------------------------------------------

.. code-block:: python

    from cubats.slide_collection.slide_collection import SlideCollection

    # Compute all possible antigen pairs within the collection
    my_collection.generate_antigen_pair_combinations(
        save_imgs = True,
        masking_mode = "pixel-level"
    )

After completion, the dual-antigen results are available in :code:`my_collection.dual_antigen_expressions`
and saved in :code:`data_dir/pixel-level_dual_antigen_expressions.csv`.

.. note::
    Co-expression analysis relies on previously quantified slides.
    Ensure quantification has been completed before running antigen combination analysis.
    Triplet combinations can be computed analogously using
    :code:`my_collection.generate_antigen_triplet_combinations(masking_mode="pixel-level")`.

.. _References:

References
==========

.. [1] Ruifrok, A., & Johnston, D. Quantification of histochemical staining by color deconvolution. Anal Quant Cytol Histol, 2001, 23.

.. [2] Varghese, F., Bukhari, A. B., Malhotra, R., & De, A. IHC Profiler: an open source plugin for the quantitative evaluation and automated scoring of immunohistochemistry images of human tissue samples. PLoS One, 2014, 9(5): e96801.
