# Standard Library
import argparse
import os
# import sys
import time
import traceback

# CuBATS
import cubats.registration as registration
import cubats.segmentation as segmentation
from cubats.slide_collection.slide_collection import SlideCollection

# # The path can also be read from a config file, etc.
# OPENSLIDE_PATH = r"C:\Users\mlser\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin"
# if hasattr(os, "add_dll_directory"):
#     with os.add_dll_directory(OPENSLIDE_PATH):
#         # Third Party
#         import openslide
# else:
#     import openslide


def main(path):
    start_time = time.time()
    # folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    # folder_to_process = folders[:30]
    for folder_name in os.listdir(path):
        src = os.path.join(path, folder_name)
        dst = r""

        if os.path.isdir(src):
            try:
                registration.register(src, dst, True, 2000)
            except Exception as e:
                error = f"Error Aligning files {path}: {str(e)}"
                traceback.print_exc()

        src = os.path.join(dst, folder_name, "registered_slides")
        if not os.path.exists(src):
            print(f"Skipping folder: {folder_name} as source can not be found.")
            continue

        he_path = ""

        for filename in os.listdir(src):
            filepath = os.path.join(src, filename)
            if (
                os.path.isfile(filepath)
                and filename.endswith(".ome.tiff")
                and "HE" in filename
            ):
                he_path = os.path.join(src, filename)
                print(f"Path ot HE: {he_path}")
                break
        dst = os.path.join(path, folder_name, "cubats")
        os.makedirs(dst, exist_ok=True)
        begin_processing = time.time()
        if not os.path.exists(os.path.join(src, folder_name, "_HE_Mask.tiff")):
            print("Mask not found: performing Tumor Segmentation")
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
            except Exception as e:
                error = f"Error Segmenting file {he_path}: {str(e)}"
                traceback.print_exc()
        try:

            collection = SlideCollection(
                collection_name=folder_name, src_dir=src, dest_dir=dst
            )
            begin_quant = time.time()
            collection.quantify_all_slides()
            end_quant = time.time()
            time_quant = (end_quant - begin_quant) / 60
            begin_dual = time.time()
            collection.get_dual_antigen_combinations()
            end_dual = time.time()
            time_dual = (end_dual - begin_dual) / 60
            begin_triple = time.time()
            collection.get_triplet_antigen_combinations()
            end_triple = time.time()
            time_triple = (end_triple - begin_triple) / 60
            end_processing = time.time()
            time_tumor = (end_processing - begin_processing) / 60
            print(f"Time segmentation: {time_seg} min")
            print(f"Time Quantification: {time_quant} min")
            print(f"Time Dual: {time_dual} min")
            print(f"Time Triple: {time_triple} min")
            print(f"Total processing: {time_tumor} min")
            with open(log_file, "a") as log:
                log.write(f"Folder: {folder_name}\n")
                log.write(f"Time Quantification: {time_quant:.4f} min\n")
                log.write(f"Time Dual: {time_dual:.4f} min\n")
                log.write(f"Time Triple: {time_triple:.4f} min\n")
                log.write(f"Total processing: {time_tumor:.4f} min\n")
                log.write("-" * 40 + "\n")
        except Exception as e:
            error = f"Error Slidecollection {path}: {str(e)}"
            traceback.print_exc(error)  # rm error

    end_time = time.time()
    total_time = (end_time - start_time) / 3600
    print(f"Processing completed in {total_time:.4f} hours")
    with open(log_file, "a") as log:
        log.write(f"Processing completed in {total_time:.4f} hours\n")


if __name__ == "__main__":
    log_file = os.path.join(
        r"H:\Notebookstuff\Sail_stuff\organoid_out", "processing_times.log"
    )
    with open(log_file, "a") as log:
        log.write(f"\nProcessing started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    parser = argparse.ArgumentParser(description="Process Tumor Set")
    parser.add_argument("path", type=str, help="Path to folder")
    args = parser.parse_args()

    if os.path.exists(args.path):
        main(args.path)
    else:
        print("Error: folder doesnt exist")
