# Standard Library
import os

# Third Party
from valis import registration

SLIDE_SRC_DIR = ""
RESULTS_DST_DIR = ""
REGISTERED_SLIDE_DEST_DIR = ""
REFERENCE_SLIDE = ""
DEFAULT_MAX_NON_RIGID_REG_SIZE = 3000
CROP = "overlap"


def register_with_ref(
    slide_src_dir,
    results_dst_dir,
    referenceSlide,
    microregistration=False,
    max_non_rigid_registartion_dim_px=DEFAULT_MAX_NON_RIGID_REG_SIZE,
):
    """
    Register the slides with a reference slide using Valis. This function automatically registers the slides and saves the registered slides in the specified directory.

    Args:
        slide_src_dir (_type_): _description_
        results_dst_dir (_type_): _description_
        referenceSlide (_type_): _description_
        microregistration (bool, optional): _description_. Defaults to False.
        max_non_rigid_registartion_dim_px (_type_, optional): _description_. Defaults to DEFAULT_MAX_NON_RIGID_REG_SIZE.
    """
    SLIDE_SRC_DIR = slide_src_dir
    RESULTS_DST_DIR = results_dst_dir
    REGISTERED_SLIDE_DEST_DIR = os.path.join(RESULTS_DST_DIR, "/registered_slides")
    REFERENCE_SLIDE = referenceSlide
    DEFAULT_MAX_NON_RIGID_REG_SIZE = max_non_rigid_registartion_dim_px

    registrar = registration.Valis(
        SLIDE_SRC_DIR, RESULTS_DST_DIR, reference_img_f=REFERENCE_SLIDE
    )
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    if microregistration:
        registrar.register_micro(
            max_non_rigid_registartion_dim_px=DEFAULT_MAX_NON_RIGID_REG_SIZE, align_to_reference=True
        )

    registrar.warp_and_save_slides(REGISTERED_SLIDE_DEST_DIR, CROP)
    registration.kill_jvm()


def register(
    slide_src_dir,
    results_dst_dir,
    microregistration=False,
    max_non_rigid_registartion_dim_px=DEFAULT_MAX_NON_RIGID_REG_SIZE,
):
    """
    Register the slides using Valis. This function automatically registers the slides and saves the registered slides in the specified directory.

    Args:
        slide_src_dir (_type_): _description_
        results_dst_dir (_type_): _description_
        microregistration (bool, optional): _description_. Defaults to False.
        max_non_rigid_registartion_dim_px (_type_, optional): _description_. Defaults to DEFAULT_MAX_NON_RIGID_REG_SIZE.
    """
    SLIDE_SRC_DIR = slide_src_dir
    RESULTS_DST_DIR = results_dst_dir
    REGISTERED_SLIDE_DEST_DIR = os.path.join(RESULTS_DST_DIR, "registered_slides")
    DEFAULT_MAX_NON_RIGID_REG_SIZE = max_non_rigid_registartion_dim_px

    registrar = registration.Valis(SLIDE_SRC_DIR, RESULTS_DST_DIR)

    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    if microregistration:
        registrar.register_micro(max_non_rigid_registartion_dim_px=DEFAULT_MAX_NON_RIGID_REG_SIZE)

    registrar.warp_and_save_slides(REGISTERED_SLIDE_DEST_DIR, crop=CROP)
    registration.kill_jvm()
