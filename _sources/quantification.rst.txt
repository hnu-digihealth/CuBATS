Quantification
==============

Module Documentation
---------------------
This module performs tile-level quantification of IHC tiles, including color deconvolution, quantification of staining intensities, H-score and the IHC-Profiler score calculation, all based on antigen-specific thresholds, and masking of tumor-tissue areas.

.. note::
    The implementation of the color-deconvolution method is based on Ruifrok & Johnston [1]_, and the pixel-intensity scoring is an adaptation of Varghese et al. IHC-Profiler algorithm [2]_.

Primary Functions
-----------------
.. autofunction:: cubats.slide_collection.tile_quantification.quantify_tile

Internal Functions
------------------
.. autofunction:: cubats.slide_collection.tile_quantification.color_deconvolution
.. autofunction:: cubats.slide_collection.tile_quantification.evaluate_staining_intensities
.. autofunction:: cubats.slide_collection.tile_quantification.calculate_percentage_and_score
.. autofunction:: cubats.slide_collection.tile_quantification.mask_tile

References
----------------
.. [1] Ruifrok, A., & Johnston, D. Quantification of histochemical staining by color deconvolution. Anal Quant Cytol Histol, 2001, 23. Available at: `<https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_ihc_color_separation.html>`_

.. [2] Varghese, F., Bukhari, A. B., Malhotra, R., & De, A. IHC Profiler: an open source plugin for the quantitative evaluation and automated scoring of immunohistochemistry images of human tissue samples. PLoS One, 2014, 9(5): e96801.
