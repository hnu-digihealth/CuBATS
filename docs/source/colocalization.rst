Co-Expression Analysis
======================

Module Documentation
---------------------
This module performs evaluation of spatial co-expression of two or three antigens based on their staining intensities within tumor-tissue areas. It calculates combined tumor coverage as well as overlapping and complementary expression of the evaluated antigens.

Primary Functions
-----------------
.. autofunction:: cubats.slide_collection.tile_colocalization.evaluate_antigen_pair_tile
.. autofunction:: cubats.slide_collection.tile_colocalization.evaluate_antigen_triplet_tile

Internal Functions
------------------
.. autofunction:: cubats.slide_collection.tile_colocalization._process_single_tile
.. autofunction:: cubats.slide_collection.tile_colocalization._process_two_tiles
.. autofunction:: cubats.slide_collection.tile_colocalization._process_three_tiles
