
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


Tile Processing
================
Module Documentation
---------------------
.. automodule:: cubats.slide_collection.tile_processing
    :exclude-members: quantify_tile, ihc_stain_separation, calculate_pixel_intensity, calculate_percentage_and_score, mask_tile, separate_stains_and_save_tiles_as_tif

Primary Functions
-----------------
.. autofunction:: cubats.slide_collection.tile_processing.quantify_tile

Internal Functions
------------------
.. autofunction:: cubats.slide_collection.tile_processing.ihc_stain_separation
.. autofunction:: cubats.slide_collection.tile_processing.calculate_pixel_intensity
.. autofunction:: cubats.slide_collection.tile_processing.calculate_percentage_and_score
.. autofunction:: cubats.slide_collection.tile_processing.mask_tile
.. autofunction:: cubats.slide_collection.tile_processing.separate_stains_and_save__tiles_as_tif

Antigen Analysis
================

Primary Functions
-----------------
.. autofunction:: cubats.slide_collection.colocalization.analyze_dual_antigen_colocalization
.. autofunction:: cubats.slide_collection.colocalization.analyze_triplet_antigen_colocalization

Internal Functions
------------------
.. automodule:: cubats.slide_collection.colocalization
    :members:
    :exclude-members: analyze_dual_antigen_colocalization, analyze_triplet_antigen_colocalization
    :undoc-members:
    :private-members:
    :show-inheritance:
