
Slide Collection
****************
Classes
========

Slide Collection
------------------
.. autoclass:: cubats.slide_collection.slide_collection.SlideCollection
    :members: __init__, generate_mask, quantify_all_slides, quantify_single_slide, get_dual_antigen_combinations, get_triplet_antigen_combinations

Slide
------
.. autoclass:: cubats.slide_collection.slide.Slide
    :members: __init__, quantify_slide, summarize_quantification_results, reconstruct_slide


Submodules
===========

Tile Processing
-----------------
.. automodule:: cubats.slide_collection.tile_processing
    :members: ihc_stain_separation, quantify_tile, calculate_pixel_intensity, calculate_score, separate_stains_and_save__tiles_as_tif

Antigen Analysis
------------------
.. automodule:: cubats.slide_collection.colocalization
    :members: compute_dual_antigen_colocalization, compute_triplet_antigen_colocalization
