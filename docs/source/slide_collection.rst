
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
Primary Functions
-----------------
.. autofunction:: cubats.slide_collection.tile_processing.quantify_tile

Internal Functions
------------------
.. automodule:: cubats.slide_collection.tile_processing
    :members:
    :exclude-members: quantify_tile
    :undoc-members:
    :private-members:
    :show-inheritance:

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
