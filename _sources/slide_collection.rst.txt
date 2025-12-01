
Slide Collection
****************

Overview
========
The :class:`SlideCollection <cubats.slide_collection.slide_collection.SlideCollection>` is the central controller of CuBATS. It organizes and manages a collection of whole-slide images (WSIs), with each WSI being implemented by the :class:`Slide <cubats.slide_collection.slide.Slide>` class. The :class:`SlideCollection` class loads and stores processing results, the status of the pipeline.
It wraps all functional processing stages of CuBATS:

1. Image Registration
2. Tumor Segmentation
3. Quantification of Staining Intensities
4. Combinatorial Analysis of Antigen Co-Expression

Each stage is implemented as a separate module and called internally by the `SlideCollection` class.

.. note::
    You can still run individual modules separately, particularly the registration and segmentation modules. However, the quantification and combinatorial analysis modules are tightly integrated into the :class:`SlideCollection` class and cannot be run independently.

Classes
========

Slide Collection
------------------
.. autoclass:: cubats.slide_collection.slide_collection.SlideCollection
    :members: __init__, register_slides, tumor_segmentation, extract_mask_tile_coordinates, quantify_all_slides, quantify_single_slide, generate_antigen_pair_combinations, generate_antigen_triplet_combinations, evaluate_antigen_pair, evaluate_antigen_triplet

Slide
------
.. autoclass:: cubats.slide_collection.slide.Slide
    :members: __init__, quantify_slide, summarize_quantification_results, reconstruct_slide
