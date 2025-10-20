# Standard Library
import glob
import os
import sys
import time
import traceback

# CuBATS
# import cubats.registration as registration
from cubats.slide_collection import segmentation as segmentation
from cubats.slide_collection.slide_collection import SlideCollection

# The path can also be read from a config file, etc.
# OPENSLIDE_PATH = r"C:\Users\mlser\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin"
# if hasattr(os, "add_dll_directory"):
#     with os.add_dll_directory(OPENSLIDE_PATH):
#         # Third Party
#         import openslide
# else:
#     import openslide


OUT_PATH = r"F:\CuBATS_out"  # Your output directory
# --- add this near the top, after OUT_PATH definition ---
KEEP_PICKLES = {
    "quantification_results.pickle",
    "dual_antigen_expressions.pickle",
    "triplet_antigen_expressions.pickle",
}


def cleanup_pickle_files(folder_dst):
    """Delete unnecessary pickle files in data/pickle while keeping the important ones."""
    pickle_dir = os.path.join(folder_dst, "data", "pickle")
    if not os.path.isdir(pickle_dir):
        return

    pickle_files = glob.glob(os.path.join(pickle_dir, "*.pickle"))
    for f in pickle_files:
        if os.path.basename(f) in KEEP_PICKLES:
            print(f"Keeping: {f}")
            continue
        try:
            os.remove(f)
            print(f"Deleted: {f}")
        except Exception as e:
            print(f"Could not delete {f}: {e}")


def main(paths):
    """
    paths: list of paths (storages) where sources are found.
    """
    start_time = time.time()

    # Collect all source folders across storages
    source_folders = set()
    for base_path in paths:
        if not os.path.exists(base_path):
            print(f"Warning: source path does not exist: {base_path}")
            continue
        for folder_name in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder_name)
            if os.path.isdir(folder_path):
                source_folders.add(folder_name)

    print(f"Found {len(source_folders)} source folders across storages.")

    # Process each source folder from all storages (if duplicates, it will process once)
    for folder_name in source_folders:
        # Find which storage contains this folder
        src = None
        for base_path in paths:
            potential_src = os.path.join(
                base_path, folder_name, "registered_slides")
            if os.path.exists(potential_src):
                src = potential_src
                break

        if src is None:
            print(
                f"Skipping {folder_name}: no registered_slides source found in storages."
            )
            continue

        dst = os.path.join(OUT_PATH, folder_name)
        if dst is None:
            print(f"Skipping {folder_name}:  not found in {OUT_PATH}.")
            continue
        # os.makedirs(dst, exist_ok=True)

        # Check if output folder already has results and skip if you want
        # Paths to expected output files in OUT_PATH/[folder_name]/data
        # data_path = os.path.join(dst, "data")
        # quant_file = os.path.join(data_path, "quantification_results.csv")
        # dual_file = os.path.join(data_path, "dual_antigen_expressions.csv")
        # triplet_file = os.path.join(data_path, "triplet_antigen_expressions.csv")

        # Skip processing if any expected result files already exist
        data_dir = os.path.join(dst, "data")
        result_files = [
            "pixel-level_quantification_results.csv",
            "pixel-level_dual_antigen_expressions.csv",
            "pixel-level_triplet_antigen_expressions.csv",
        ]
        all_exist = all(os.path.exists(os.path.join(data_dir, f))
                        for f in result_files)
        if all_exist:
            print(f"Skipping {folder_name}: all result files already exist.")
            continue

        print(f"Processing folder: {folder_name}")
        begin_processing = time.time()

        try:
            he_path = ""
            for filename in os.listdir(src):
                filepath = os.path.join(src, filename)
                if (
                    os.path.isfile(filepath)
                    and filename.endswith(".ome.tiff")
                    and "HE" in filename
                ):
                    he_path = os.path.join(src, filename)
                    print(f"Path to HE: {he_path}")
                    break

            # Check if mask file exists before running segmentation
            mask_filename = f"{folder_name}_HE_mask.tiff"
            mask_path = os.path.join(src, mask_filename)
            if not os.path.exists(mask_path) and he_path:
                print(f"Mask file found: {mask_path}, running segmentation.")
                try:
                    begin_seg = time.time()
                    model_path = (
                        r"C:\Users\mlser\Desktop\ml_model\seg_mod_256_2023-02-15.onnx"
                    )
                    tile_size = (512, 512)
                    segmentation.run_tumor_segmentation(
                        input_path=he_path,
                        model_path=model_path,
                        tile_size=tile_size,
                        output_path=None,
                        normalization=False,
                        inversion=False,
                        plot_results=False,
                    )
                    end_seg = time.time()
                    time_seg = (end_seg - begin_seg) / 60
                    print(f"Segmentation completed in {time_seg:.2f} minutes")
                except Exception as e:
                    error = f"Error segmenting file {he_path}: {str(e)}"
                    print(error)
                    traceback.print_exc()

            collection = SlideCollection(
                collection_name=folder_name,
                src_dir=src,
                dest_dir=dst,
                path_antigen_profiles=r"C:\Users\mlser\Desktop\CuBATS\src\cubats\assets\antigen_profiles.json",
            )
            begin_quant = time.time()
            collection.quantify_all_slides(masking_mode="pixel-level")
            end_quant = time.time()
            time_quant = (end_quant - begin_quant) / 60
            print(f"Quantification took {time_quant:.2f} minutes")

            begin_dual = time.time()
            collection.generate_antigen_pair_combinations(
                masking_mode="pixel-level")
            end_dual = time.time()
            time_dual = (end_dual - begin_dual) / 60
            print(f"Dual antigen took {time_dual:.2f} minutes")

            begin_triple = time.time()
            collection.generate_antigen_triplet_combinations(
                masking_mode="pixel-level")
            end_triple = time.time()
            time_triple = (end_triple - begin_triple) / 60
            print(f"Triple antigen took {time_triple:.2f} minutes")
            end_processing = time.time()
            time_tumor = (end_processing - begin_processing) / 60
            print(f"Total processing: {time_tumor} min")
            # Write logs
            with open(log_file, "a") as log:
                log.write(f"Folder: {folder_name}\n")
                log.write(f"Time Quantification: {time_quant:.4f} min\n")
                log.write(f"Time Dual: {time_dual:.4f} min\n")
                log.write(f"Time Triple: {time_triple:.4f} min\n")
                log.write(f"Total processing: {time_tumor:.4f} min\n")
                log.write("-" * 40 + "\n")

            cleanup_pickle_files(dst)
        except Exception as e:
            print(f"Error processing {folder_name}: {e}")
            traceback.print_exc()

    # Now check for any output folders without matching source folders
    output_folders = [
        f for f in os.listdir(OUT_PATH) if os.path.isdir(os.path.join(OUT_PATH, f))
    ]
    for out_folder in output_folders:
        if out_folder not in source_folders:
            print(
                f"Output folder '{out_folder}' has no matching source folder in provided storages."
            )

    total_time = (time.time() - start_time) / 3600
    print(f"Processing completed in {total_time:.2f} hours")


if __name__ == "__main__":
    # Standard Library
    import argparse

    log_file = os.path.join(r"F:\CuBATS_out", "processing_times_1.log")
    with open(log_file, "a") as log:
        log.write(
            f"\nProcessing started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    parser = argparse.ArgumentParser(description="Process Tumor Set")
    parser.add_argument(
        "paths",
        nargs="+",
        type=str,
        help="One or more paths to ," " storages (folders containing sample folders)",
    )
    args = parser.parse_args()

    for p in args.paths:
        if not os.path.exists(p):
            print(f"Error: source folder does not exist: {p}")
            sys.exit(1)

    # Create output folder if not exists
    os.makedirs(OUT_PATH, exist_ok=True)

    main(args.paths)
