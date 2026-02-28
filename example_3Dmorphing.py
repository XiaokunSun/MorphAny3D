import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils.morphing_utils import *

SEED = 0
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

src_img_path_list = []
tar_img_path_list = []

for tmp_name in [
    ["Godzilla.png", "typical_humanoid_dragonborn.png"],
    ["typical_vehicle_pirate_ship.png", "typical_vehicle_biplane.png"],
    ["Pigsy.png", "Sun_Wukong.png"],
    ["Bull_Demon_King.png", "head.png"],
    ["Big_Mesh_red.png", "Christmas_tree.png"]
    ]:
    src_img_path_list.append(f"./assets/example_morphing/{tmp_name[0]}")
    tar_img_path_list.append(f"./assets/example_morphing/{tmp_name[1]}")

save_dir_path = "./outputs"

for idx in range(len(src_img_path_list)):
    src_img_path = src_img_path_list[idx]
    tar_img_path = tar_img_path_list[idx]
    src_img = Image.open(src_img_path)
    tar_img = Image.open(tar_img_path)
    src_name = os.path.basename(src_img_path).split(".")[0]
    tar_name = os.path.basename(tar_img_path).split(".")[0]
    src_save_path = os.path.join(save_dir_path, "cache", src_name)
    tar_save_path = os.path.join(save_dir_path, "cache", tar_name)
    src_save_cache_path = os.path.join(src_save_path, "cache")
    tar_save_cache_path = os.path.join(tar_save_path, "cache")
    os.makedirs(src_save_path, exist_ok=True)
    os.makedirs(tar_save_path, exist_ok=True)
    os.makedirs(src_save_cache_path, exist_ok=True)
    os.makedirs(tar_save_cache_path, exist_ok=True)
    morphing_params = {"save_cache_path": src_save_cache_path, "init_morphing_flag": False, "ss_mca_flag": False, "slat_mca_flag": False, "ss_tfsa_flag": False, "slat_tfsa_flag": False, "oc_flag": False}
    if not os.path.exists(f"{src_save_cache_path}/slat_init.pt"):
        run_morphing_cache(pipeline, src_img, tar_img, morphing_params, SEED, src_save_path, src_name)
    morphing_params = {"save_cache_path": tar_save_cache_path, "init_morphing_flag": False, "ss_mca_flag": False, "slat_mca_flag": False, "ss_tfsa_flag": False, "slat_tfsa_flag": False, "oc_flag": False}
    if not os.path.exists(f"{tar_save_cache_path}/slat_init.pt"):
        run_morphing_cache(pipeline, tar_img, src_img, morphing_params, SEED, tar_save_path, tar_name)

    name = src_name + "+" + tar_name

    morphing_params = {"morphing_num": 50, "src_load_cache_path": src_save_cache_path, "tar_load_cache_path": tar_save_cache_path, "init_morphing_flag": False, "ss_mca_flag": True, "slat_mca_flag": True, "ss_tfsa_flag": True, "slat_tfsa_flag": True, "oc_flag": True} # When you observe orientation jumps, set oc_flag to True
    save_path = os.path.join(save_dir_path, "3Dmorphing", name)
    os.makedirs(save_path, exist_ok=True)
    save_cache_path = os.path.join(save_path, "cache")
    os.makedirs(save_cache_path, exist_ok=True)
    morphing_params["save_cache_path"] = save_cache_path
    run_morphing(pipeline, src_img, tar_img, morphing_params, SEED, save_path, name)