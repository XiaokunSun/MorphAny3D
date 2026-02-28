import numpy as np
import torch
from ..modules.sparse.basic import SparseTensor
import torch.nn.functional as F
import random
import os
from glob import glob
import imageio
from ..utils import render_utils

def unique_rows_with_mask(x: torch.Tensor):
    seen = set()
    mask = []
    for row in x.tolist():
        tup = tuple(row)
        if tup not in seen:
            seen.add(tup)
            mask.append(True)   # 保留
        else:
            mask.append(False)  # 丢弃
    mask_tensor = torch.tensor(mask, dtype=torch.bool).to(x.device)
    return mask_tensor

def cal_cossim_matrix(feats1, feats2):
    feats1 = feats1.reshape(feats1.shape[0], -1)
    feats2 = feats2.reshape(feats2.shape[0], -1)
    feats1 = F.normalize(feats1, dim=-1)
    feats2 = F.normalize(feats2, dim=-1)
    cossim_matrix = torch.matmul(feats1, feats2.t())
    return cossim_matrix

def cal_eucdist_matrix(tensor1, tensor2):
    XX = (tensor1 * tensor1).sum(dim=1, keepdim=True)
    YY = (tensor2 * tensor2).sum(dim=1, keepdim=True).T
    D2 = XX + YY - 2.0 * (tensor1 @ tensor2.T)
    D2.clamp_(min=0)                            
    return torch.sqrt(D2)

def slat_interp(slat1, slat2, alpha: float, mapping_mode="order", interp_mode="linear", unique_flag=True, indices=None):
    if mapping_mode == "order":
        if slat1.feats.shape[0] <= slat2.feats.shape[0]:
            indices = torch.arange(slat1.feats.shape[0], device=slat1.feats.device)
        else:
            tmp1 = slat1.feats.shape[0] // slat2.feats.shape[0]
            tmp2 = slat1.feats.shape[0] % slat2.feats.shape[0]
            indices = [torch.arange(slat2.feats.shape[0], device=slat1.feats.device)] * tmp1 + [torch.arange(tmp2, device=slat1.feats.device)]
            indices = torch.cat(indices, dim=0)
    elif mapping_mode == "cossim":
        cossim_matrix = cal_cossim_matrix(slat1.feats, slat2.feats)
        indices = torch.argmax(cossim_matrix, dim=1)
    elif mapping_mode == "eucdist":
        eucdist_matrix = cal_eucdist_matrix(slat1.coords[:, 1:] / slat1.coords.max(), slat2.coords[:, 1:] / slat2.coords.max())
        indices = torch.argmin(eucdist_matrix, dim=1)
    elif mapping_mode == "hungarian":
        pass

    if interp_mode == "linear":
        interp_feats = alpha * slat1.feats + (1 - alpha) * slat2.feats[indices]
        interp_coords = torch.round((alpha * slat1.coords + (1 - alpha) * slat2.coords[indices])).int()

    if unique_flag:
        mask = unique_rows_with_mask(interp_coords)
        interp_feats = interp_feats[mask]
        interp_coords = interp_coords[mask]
    return SparseTensor(feats=interp_feats, coords=interp_coords)

def slat_conc(slat1, slat2, dim=0, unique_flag=False):
    conc_feats = torch.cat([slat1.feats, slat2.feats], dim=dim)
    conc_coords = torch.cat([slat1.coords, slat2.coords], dim=dim)

    if unique_flag:
        mask = unique_rows_with_mask(conc_coords)
        conc_feats = conc_feats[mask]
        conc_coords = conc_coords[mask]
    return SparseTensor(feats=conc_feats, coords=conc_coords)

def feature_interp(tensor1, tensor2, alpha: float, mapping_mode: str = "order", interp_mode: str = "linear", indices=None):
    if mapping_mode == "order":
        pass
    elif mapping_mode == "cossim":
        cossim_matrix = cal_cossim_matrix(tensor1, tensor2)
        indices = torch.argmax(cossim_matrix, dim=1)
        tensor2 = tensor2[indices]
    elif mapping_mode == "hungarian":
        tensor2 = tensor2[indices]

    if interp_mode == "linear":
        return alpha * tensor1 + (1 - alpha) * tensor2
    elif interp_mode == "slerp":
        a = tensor1
        b = tensor2
        an = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        bn = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        dot = (an * bn).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

        close = (dot.abs() > 0.9999).squeeze(-1)
        theta = torch.acos(dot) * (1 - alpha)

        rel = (bn - dot * an)
        rel = rel / rel.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        slerp_dir = an * torch.cos(theta) + rel * torch.sin(theta)

        lin_dir = alpha * an + (1 - alpha) * bn
        lin_dir = lin_dir / lin_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        slerp_dir = torch.where(close.unsqueeze(-1), lin_dir, slerp_dir)

        ra = a.norm(dim=-1, keepdim=True)
        rb = b.norm(dim=-1, keepdim=True)
        r  = alpha * ra + (1 - alpha) * rb
        return slerp_dir * r
    else:
        raise ValueError("don't support this mode")

def find_surface_voxels(voxels):
    voxels_x1 = torch.zeros_like(voxels).bool()
    voxels_x2 = torch.zeros_like(voxels).bool()
    voxels_y1 = torch.zeros_like(voxels).bool()
    voxels_y2 = torch.zeros_like(voxels).bool()
    voxels_z1 = torch.zeros_like(voxels).bool()
    voxels_z2 = torch.zeros_like(voxels).bool()

    voxels_x1[:-1, :, :] = voxels[1:, :, :]
    voxels_x2[1:, :, :] = voxels[:-1, :, :]
    voxels_y1[:, :-1, :] = voxels[:, 1:, :]
    voxels_y2[:, 1:, :] = voxels[:, :-1, :]
    voxels_z1[:, :, :-1] = voxels[:, :, 1:]
    voxels_z2[:, :, 1:] = voxels[:, :, :-1]

    surface_voxels = voxels & (~(voxels_x1 & voxels_x2 & voxels_y1 & voxels_y2 & voxels_z1 & voxels_z2))
    return surface_voxels

def rotate_pc(
    points: torch.Tensor,
    angle: float,
    axis: str = "z"
) -> torch.Tensor:
    single = (points.dim() == 2)
    if single:
        points = points.unsqueeze(0) 
    B, N, D = points.shape
    assert D == 3

    device, dtype = points.device, points.dtype
    ang = torch.as_tensor(angle, device=device, dtype=dtype)
    ang = ang * torch.pi / 180.0
    if ang.dim() == 0:
        ang = ang.expand(B)
    else:
        assert ang.shape == (B,)

    c = torch.cos(ang)
    s = torch.sin(ang)

    if axis in ("x", 0):
        R = torch.stack([
            torch.stack([torch.ones_like(c), torch.zeros_like(c), torch.zeros_like(c)], dim=-1),
            torch.stack([torch.zeros_like(c), c, -s], dim=-1),
            torch.stack([torch.zeros_like(c), s,  c], dim=-1),
        ], dim=-2) 
    elif axis in ("y", 1):
        R = torch.stack([
            torch.stack([ c, torch.zeros_like(c), s], dim=-1),
            torch.stack([torch.zeros_like(c), torch.ones_like(c), torch.zeros_like(c)], dim=-1),
            torch.stack([-s, torch.zeros_like(c), c], dim=-1),
        ], dim=-2)
    elif axis in ("z", 2):
        R = torch.stack([
            torch.stack([c, -s, torch.zeros_like(c)], dim=-1),
            torch.stack([s,  c, torch.zeros_like(c)], dim=-1),
            torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)], dim=-1),
        ], dim=-2)
    else:
        raise ValueError("axis must be 'x', 'y' or 'z'")

    C = torch.zeros(B, 1, 3, device=device, dtype=dtype)

    P = points - C
    Prot = torch.matmul(P, R.transpose(1, 2)) + C

    return Prot.squeeze(0) if single else Prot

def save_pc(pc, save_path, color=None):
    obj_str = ""
    for i in range(pc.shape[0]):
        if color is None:
            obj_str += f"v {pc[i][0]} {pc[i][1]} {pc[i][2]}"
        else:
            obj_str += f"v {pc[i][0]} {pc[i][1]} {pc[i][2]} {color[i][0]} {color[i][1]} {color[i][2]}"
        obj_str += "\n"
    obj_str += "\n"
    with open(save_path, "w") as f:
        f.write(obj_str)
    return

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def run_morphing_cache(pipeline, src_img, tar_img, morphing_params, seed, save_path, name, bg_color=(1, 1, 1)):
    seed_everything(seed)
    with torch.no_grad():
        outputs = pipeline.run_morphing(
            src_img=src_img,
            tar_img=tar_img,
            morphing_params=morphing_params,
            seed=seed
        )
    gs_video = render_utils.render_rot_video(outputs['gaussian'][0], bg_color=bg_color)['color']
    imageio.mimsave(f"{save_path}/{name}.mp4", gs_video, fps=30)
    mesh_video = render_utils.render_rot_video(outputs['mesh'][0], bg_color=bg_color)['normal']
    imageio.mimsave(f"{save_path}/{name}_normal.mp4", mesh_video, fps=30)

def run_morphing(pipeline, src_img, tar_img, morphing_params, seed, save_path, name, bg_color=(1, 1, 1)):
    seed_everything(seed)
    morphing_params["rm_cache"] = True
    if morphing_params["rm_cache"]:
        files = glob(f"{morphing_params['save_cache_path']}/*")
        for f in files:
            os.remove(f)

    if os.path.exists(f"{save_path}/morphing_{name}.mp4"):
        print(f"Skip existing {name}")
    else:
        gs_video_list = []
        mesh_video_list = []
        alpha_array = np.linspace(1, 0, morphing_params["morphing_num"])

        for morphing_idx in range(1, morphing_params["morphing_num"] - 1):
            morphing_params["alpha"] = alpha_array[morphing_idx]
            morphing_params["morphing_idx"] = morphing_idx
            morphing_params["tfsa_cache_idx"] = morphing_idx - 1
            morphing_params["tfsa_alpha"] = 0.8

            with torch.no_grad():
                outputs = pipeline.run_morphing(
                    src_img=src_img,
                    tar_img=tar_img,
                    morphing_params=morphing_params,
                    seed=seed
                )

            gs_video_list.append(np.stack(render_utils.render_rot_video(outputs['gaussian'][0], bg_color=bg_color)['color'], axis=0))
            mesh_video_list.append(np.stack(render_utils.render_rot_video(outputs['mesh'][0], bg_color=bg_color)['normal'], axis=0))

            if morphing_params["rm_cache"]:
                files = glob(f"{morphing_params['save_cache_path']}/ss_sa_morphing{morphing_params['tfsa_cache_idx']}_*") + glob(f"{morphing_params['save_cache_path']}/slat_sa_morphing{morphing_params['tfsa_cache_idx']}_*")
                for f in files:
                    os.remove(f)

        if morphing_params["rm_cache"]:
            files = glob(f"{morphing_params['save_cache_path']}/*")
            for f in files:
                os.remove(f)

        gs_video = np.stack(gs_video_list, axis=0)
        mesh_video = np.stack(mesh_video_list, axis=0)

        view_idx = [0, 20, 40, 60, 80, 100]
        morphing_gs_video = np.concatenate(np.transpose(gs_video[:, view_idx, ...], (1, 0, 2, 3, 4)), axis=2)
        morphing_mesh_video = np.concatenate(np.transpose(mesh_video[:, view_idx, ...], (1, 0, 2, 3, 4)), axis=2)
        morphing_idx = np.arange(0, morphing_params["morphing_num"]-2, (morphing_params["morphing_num"]-2)//4).tolist()
        if len(morphing_idx) < 5:
            morphing_idx.append(morphing_params["morphing_num"]-3)
        show_gs_video = np.concatenate(gs_video[morphing_idx, ...], axis=2)
        show_mesh_video = np.concatenate(mesh_video[morphing_idx, ...], axis=2)

        imageio.mimsave(f"{save_path}/morphing_{name}.mp4", morphing_gs_video, fps=(morphing_params["morphing_num"]-2)//2)
        imageio.mimsave(f"{save_path}/morphing_{name}_normal.mp4", morphing_mesh_video, fps=(morphing_params["morphing_num"]-2)//2)
        imageio.mimsave(f"{save_path}/show_{name}.mp4", show_gs_video, fps=30)
        imageio.mimsave(f"{save_path}/show_{name}_normal.mp4", show_mesh_video, fps=30)