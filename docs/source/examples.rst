.. _Examples:

Examples
********

.. important::
    Depending on the size of your data, make sure that your system has enough memory available to store the data, as well as enough RAM to process the amount of data. We recommend a system with at least 30 GB of RAM or using a dedicated server.

.. _Image Registration:

Image Registration
==================

.. important::
    Image registration and alignment is performed using the framework `VALIS <https://github.com/MathOnco/valis>`_. Please make sure you have installed the required dependencies. For more information check out the CuBATS :ref:`Installation` or the `VALIS documentation <https://valis.readthedocs.io/en/latest/index.html>`_. Please note that provided class facilitates registration by predefining some of the registration parameters. For more individualized registration or if you are not satisfied with the registration results, we recommend checking out the `VALIS documentation <https://valis.readthedocs.io/en/latest/index.html>`_ on how to adjust the registration parameters to your needs.

This example demonstrates how to register a set of slides in CuBATS followed by a subsequent alignment of the slides. Registration and alignment can be done towards a previously selected reference slide or towards a slide that is automatically selected by the VALIS registration algorithm. While in some cases it might be beneficial to register the slides towards a reference slide, most use cases during development have shown more accurate results when registering the slides without a reference.
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
    reference_slide = "path/to/reference_slide"

    # Register the slides towards a reference slide
    cubats.registration.register(slide_src_dir, results_dst_dir, microregistration=True,
    max_non_rigid_registartion_dim_px=2000)

After registration the results will be saved inside the :code:`results_dst_dir/registered_slides` directory. The created result_dest_dir also contains addition information and visualizations of the registration process. Except for the registered and aligned slides, the rest of the directories created by VALIS is irrelevant for CuBATS. If would like more information on the content of these directories please also refer to the `VALIS documentation <https://valis.readthedocs.io/en/latest/index.html>`_.

.. _Image Segmentation:

Image Segmentation
==================



.. _Initialize SlideCollection:

Initialize SlideCollection
==========================


.. _Quantify SlideCollection:

Quantify SlideCollection
========================


.. _Analysis:

Analysis
========
