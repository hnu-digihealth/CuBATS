Segmentation
=============

Module Documentation
--------------------
This module handles tumor segmentation of HE-stained whole-slide images (WSIs) using configurable segmentation models in ONNX format. It processes WSIs by dividing them into tiles, applying the segmentation model to each tile, and reconstructing the full segmentation mask for the entire WSI.

Primary Functions
------------------
.. autofunction:: cubats.slide_collection.segmentation.run_tumor_segmentation

Internal Functions
-------------------
.. autofunction:: cubats.slide_collection.segmentation._segment_file
.. autofunction:: cubats.slide_collection.segmentation._segment_tile
.. autofunction:: cubats.slide_collection.segmentation._plot_segmentation_on_tissue
