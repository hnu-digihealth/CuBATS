.. _Examples:

Examples
********
The following examples demonstrate a full walkthrough of the CuBATS pipeline and are meant as a guide to help you get
started with CuBATS. The examples cover the different pipeline steps starting with image registration and segmentation
of the tumor mask, followed by the analysis of the slide collection.
An exemplary execution of the pipeline can be found as a Jupyter notebook in the `CuBATS/examples` directory.

.. important::
    Depending on the size of your data, make sure that your system has enough memory available to store the data, as well as enough RAM to process the amount of data. We recommend a system with at least 30 GB of RAM or using a dedicated server.

.. _Image Registration:

Image Registration
==================

.. important::
    Image registration and alignment is performed using the framework `VALIS <https://github.com/MathOnco/valis>`_. Please make sure you have installed the required dependencies. For more information check out the CuBATS :ref:`Installation` or the `VALIS documentation <https://valis.readthedocs.io/en/latest/index.html>`_. Please note that provided class facilitates registration by predefining some of the registration parameters. For more individualized registration or if you are not satisfied with the registration results, we recommend checking out the `VALIS documentation <https://valis.readthedocs.io/en/latest/index.html>`_ on how to adjust the registration parameters to your needs.

This example demonstrates how to register a set of Whole-Slide-Images (WSIs) in CuBATS followed by a subsequent alignment of the slides. Registration and alignment can be done towards a previously selected reference slide or towards a slide that is automatically selected by the VALIS registration algorithm. While in some cases it might be beneficial to register the slides towards a reference slide, most use cases during development have shown more accurate results when registering the slides without a reference.
By default, registration includes a rigid registration followed by a non-rigid registration. Optionally, the user can choose to perform an additional high-resolution micro-registration by setting the parameter :code:`microregistration = True`. Using microregistration is recommended for large files with high resolution to achieve better registration results.

`Example 1`_ shows how to register a set of slides towards a reference slide. `Example 2`_ demonstrates how to register a set of slides without a reference. In both examples, the slides to be registered need to be located in :code:`/path/to/slides`. Following registration the slides will automatically be warped and saved into the designated destination directory :code:`/path/to/registered_slides`. By default the parameter :code:`max_non_rigid_registartion_dim_px` is set to 3000 to ensure a high-resolution registration. However, this parameter can be adjusted to the user's needs, as shown in `Example 2`_. For more information on how to adjust the parameter checkout the `VALIS documentation <https://valis.readthedocs.io/en/latest/index.html>`_.

.. _Example 1:

Example 1: Image Registration With Reference Slide
--------------------------------------------------

.. code-block:: python

    from cubats import registration
    # Paths to the slides and the destination directory
    slide_src_dir = "path/to/slides"
    results_dst_dir = "./registration_example"
    registered_slide_dest_dir = "./registration_example/registrered_slides"
    reference_slide = "path/to/reference_slide"

    # Register the slides towards a reference slide
    cubats.registration.register_with_ref(slide_src_dir, results_dst_dir, reference_slide,
    microregistration=True)

.. _Example 2:

Example 2: Image Registration Without Reference Slide
-----------------------------------------------------

.. code-block:: python

    from cubats import registration
    # Paths to the slides and the destination directory
    slide_src_dir = "path/to/slides"
    results_dst_dir = "./registration_example"
    registered_slide_dest_dir = "./registration_example/registrered_slides"

    # Register the slides without a reference slide, max_non_rigid_registartion_dim_px is set to 2000
    cubats.registration.register(slide_src_dir, results_dst_dir, microregistration=True,
    max_non_rigid_registartion_dim_px=2000)

After registration the results will be saved inside the :code:`results_dst_dir/registered_slides` directory. The created result_dest_dir also contains addition information and visualizations of the registration process. Except for the registered and aligned slides, the rest of the directories created by VALIS are irrelevant for CuBATS. If would like more information on the content of these directories please also refer to the `VALIS documentation <https://valis.readthedocs.io/en/latest/index.html>`_.

.. _Image Segmentation:

Tumor Segmentation
==================

.. important::
    The correct segmentation of tumor tissue in the WSI is highly dependent on the provided segmentation model.
    Depending on how the applied model was trained, different segmentation parameters, such as :code:`normalization` or :code:`inversion` may need to be adjusted to ensure the right results. Additionally, the quality of the segmentation results depends on the training of the model. If you are not satisfied with the segmentation results, we recommend adapting the segmentation parameters. If this does not help check if the model's training satisfies your quality expectations.

`Example 3`_ demonstrates how to run the tumor segmentation using the :code:`cubats.segemtentation.run_tumor_segmentation`. The function applies a provided segmentation model onto the provided HE-stained WSI and saves a binary tumor mask of the segmentation results as .TIFF. Tumor segmentation requires a HE-stained slide, as this staining allows a better distinction between tumor and non-tumor tissue than other stains. The input image should be located in :code:`input_path`. It is possible to both pass either a single slide or a directory containing multiple slides. Note, that when passing multiple slides, the passed directory should only contain HE-stained WSIs as segmetentation on other stains will not work as expected.

Next, a segmentation model needs to be passed as `.ONNX` file via the paramameter :code:`model_path`. Additionally, :code:`tile_size` needs to be provided as tuple (i.e. (512,512)). Optionally, further parameters can be passed. A specific output directory can be declared using the :code:`output_path` parameter (as is the case in `Example 3`_). If no :code:`output_path` is provided the results will be saved inside the input folder, regardless of whether a single file or directory was passed. Model-specific parameters are :code:`normalization`, which applies a Reinhard normalization to the input slide before segmentation, and :code:`inversion`, which inverts the binary mask after segmentation. Both parameters are set to `False` by default. Lastly, the parameter :code:`plot_results` provides the creation of a .PNG thumbnail containing a plot of the tumor mask onto the original image.

Some segmentation models may have a different input size than the selected tile size. The function will automatically apply resizing to the input tile before segmentation as well as resizing the segmented tile back to the original size.

.. _Example 3:

Example 3: Tumor Segmentation
------------------------------

.. code-block:: python

    from cubats.segmentation import run_tumor_segmentation

    # Paths to the input WSI, model, and output directory
    input_path = "path/to/wsi_file.tif"
    model_path = "path/to/model.onnx"
    output_path = "path/to/output_dir"

    # Run the tumor segmentation pipeline
    run_tumor_segmentation(
        input_path=input_path,
        model_path=model_path,
        tile_size= (512, 512),
        output_path=output_path,
        normalization=False,
        inversion=False,
        plot_results=True
    )


.. _Initialize SlideCollection:

Initialize SlideCollection
==========================



.. _Quantify SlideCollection:

Quantify SlideCollection
========================


.. _Analysis:

Analysis
========
