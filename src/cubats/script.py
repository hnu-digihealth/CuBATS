# Standard Library
import argparse
import os
import time
import traceback

# CuBATS
import cubats.segmentation as segmentation
from cubats.slide_collection.slide_collection import SlideCollection


def main(path):
    start_time = time.time()
    # folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    # folder_to_process = folders[:30]
    for folder_name in os.listdir(path):
        # src = os.path.join(path, folder_name)
        # # dst = r"G:\CuBATS"
        # reg_dst = os.path.join(dst, folder_name, "registered_slides")
        # he_name = ""
        # for filename in os.listdir(src):
        #     filepath = os.path.join(src, filename)
        #     if (
        #         os.path.isfile(filepath)
        #         and filename.endswith(".mrxs")
        #         and "HE" in filename
        #     ):
        #         he_name = os.path.splitext(filename)[0]
        #         he_new = f"{he_name}.ome.tiff"
        #         print(f"HE_name ome: {he_new}")
        #         break

        # if os.path.isdir(src):
        #     try:
        #         registrar = valis.registration.Valis(src_dir=src, dst_dir=dst)
        #         rigid, non_rigid, error_df = registrar.register()
        #         registrar.register_micro(max_non_rigid_registration_dim_px=2000)
        #         registrar.warp_and_save_slides(reg_dst, crop="overlap")
        #     except Exception as e:
        #         error = f"Error Aligning files {path}: {str(e)}"
        #         traceback.print_exc()
        src = os.path.join(path, folder_name, "registered_slides")
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
        # dst = os.path.join(path, folder_name, "cubats")
        dst = os.path.join(r"F:\CuBATS_out", folder_name)
        os.makedirs(dst, exist_ok=True)
        try:
            model_path = r"C:\Users\mlser\Desktop\ml_model\seg_mod_256_2023-02-15.onnx"
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
        except Exception as e:
            error = f"Error Segmenting file {he_path}: {str(e)}"
            traceback.print_exc()
        try:

            collection = SlideCollection(
                collection_name=folder_name, src_dir=src, dest_dir=dst
            )
            collection.quantify_all_slides()
            collection.get_dual_antigen_combinations()
            collection.get_triplet_antigen_combinations()
        except Exception as e:
            error = f"Error Slidecollection {path}: {str(e)}"
            traceback.print_exc(error)  # rm error
    end_time = time.time()
    print(f"processing completed in {(end_time - start_time)/3600} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Tumor Set")
    parser.add_argument("path", type=str, help="Path to folder")
    args = parser.parse_args()

    if os.path.exists(args.path):
        main(args.path)
    else:
        print("Error: folder doesnt exist")
