# Standard Library
import os
from time import time

# Third Party
import numpy as np
from valis import registration

SLIDE_SRC_DIR = ""
RESULTS_DST_DIR = ""
REGISTERED_SLIDE_DEST_DIR = ""
REFERENCE_SLIDE = ""
# TODO test and change to lower maybe --> max picture size when using micro registration
DEFAULT_MAX_NON_RIGID_REG_SIZE = 2000
CROP = "overlap"


def register_with_ref(
    slide_src_dir,
    results_dst_dir,
    referenceSlide,
    microregistration=False,
    max_non_rigid_registration_dim_px=DEFAULT_MAX_NON_RIGID_REG_SIZE,
):
    """
    Register the slides with a reference slide using Valis. This function automatically registers the slides and saves
    the registered slides in the specified directory.

    Args:
        slide_src_dir (_type_): _description_
        results_dst_dir (_type_): _description_
        referenceSlide (_type_): _description_
        microregistration (bool, optional): _description_. Defaults to False.
        max_non_rigid_registration_dim_px (_type_, optional): _description_. Defaults to DEFAULT_MAX_NON_RIGID_REG_SIZE
    """
    # Input validation
    if not isinstance(slide_src_dir, str) or not os.path.exists(slide_src_dir):
        raise ValueError("Invalid or non-existent source directory")
    if not isinstance(results_dst_dir, str):
        raise ValueError("Invalid destination directory")
    if not isinstance(referenceSlide, str) or not os.path.exists(referenceSlide):
        raise ValueError("Invalid or non-existent reference slide")
    if not isinstance(microregistration, bool):
        raise ValueError("microregistration must be a boolean")
    if not isinstance(max_non_rigid_registration_dim_px, int):
        raise ValueError(
            "max_non_rigid_registartion_dim_px must be an integer")

    registered_slides_dst = os.path.join(results_dst_dir, "registered_slides")

    registrar = registration.Valis(
        slide_src_dir, results_dst_dir, reference_img_f=referenceSlide
    )
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()
    if microregistration:
        registrar.register_micro(
            max_non_rigid_registration_dim_px=max_non_rigid_registration_dim_px,
            align_to_reference=True,
        )
    registrar.warp_and_save_slides(registered_slides_dst, crop=CROP)
    registration.kill_jvm()


def register(
    slide_src_dir,
    results_dst_dir,
    microregistration=False,
    max_non_rigid_registration_dim_px=DEFAULT_MAX_NON_RIGID_REG_SIZE,
):
    """
    Register the slides using Valis. This function automatically registers the slides and saves the registered slides
    in the specified directory.

    Args:
        slide_src_dir (_type_): _description_
        results_dst_dir (_type_): _description_
        microregistration (bool, optional): _description_. Defaults to False.
        max_non_rigid_registration_dim_px (_type_, optional): _description_. Defaults to DEFAULT_MAX_NON_RIGID_REG_SIZE
    """
    registered_slides_dst = os.path.join(results_dst_dir, "registered_slides")

    registrar = registration.Valis(slide_src_dir, results_dst_dir)

    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    if microregistration:
        registrar.register_micro(
            max_non_rigid_registartion_dim_px=max_non_rigid_registration_dim_px
        )
    registrar.warp_and_save_slides(registered_slides_dst, crop=CROP)
    registration.kill_jvm()


def high_resolution_alignement(slide_src_dir, results_dst_dir, micro_reg_fraction):
    """
    Performs high resolution alignment. TODO not functional yet. jupyter crashes
    """
    # Perform high resolution rigid registration using the MicroRigidRegistrar
    start = time()
    REGISTERED_SLIDE_DEST_DIR = os.path.join(
        results_dst_dir, "registered_slides")
    registrar = registration.Valis(slide_src_dir, results_dst_dir)
    # , micro_rigid_registrar_cls=MicroRigidRegistrar
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    # Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration
    # on an image that is 25% full resolution.
    img_dims = np.array(
        [
            slide_obj.slide_dimensions_wh[0]
            for slide_obj in registrar.slide_dict.values()
        ]
    )
    min_max_size = np.min([np.max(d) for d in img_dims])
    img_areas = [np.multiply(*d) for d in img_dims]
    max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])
    micro_reg_size = np.floor(min_max_size * micro_reg_fraction).astype(int)
    print(micro_reg_size)
    # Perform high resolution non-rigid registration using 25% full resolution
    micro_reg, micro_error = registrar.register_micro(
        max_non_rigid_registration_dim_px=micro_reg_size
    )
    registrar.warp_and_save_slides(REGISTERED_SLIDE_DEST_DIR, crop=CROP)
    registration.kill_jvm()
    end = time()
    print(f"High-resolution alignement completed in {end-start:.2f} seconds")
