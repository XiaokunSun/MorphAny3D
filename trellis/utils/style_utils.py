import os
import numpy as np
import torch
from ..modules.sparse.basic import SparseTensor
from PIL import Image
import cv2
import random


def calc_mean_std(feat, eps=1e-5):
    dims = list(range(len(feat.shape) - 1))
    feat_var = feat.var(dim=dims) + eps
    feat_std = feat_var.sqrt()
    feat_mean = feat.mean(dim=dims)
    return feat_mean, feat_std

def adain(content_feat, style_feat):
    if isinstance(content_feat, SparseTensor) and isinstance(style_feat, SparseTensor):
        assert (content_feat.feats.size()[-1] == style_feat.feats.size()[-1])
        size = content_feat.feats.size()
        style_mean, style_std = calc_mean_std(style_feat.feats)
        content_mean, content_std = calc_mean_std(content_feat.feats)
        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)
    elif isinstance(content_feat, torch.Tensor) and isinstance(style_feat, torch.Tensor):
        assert (content_feat.size()[-1] == style_feat.size()[-1])
        size = content_feat.size()
        style_mean, style_std = calc_mean_std(style_feat)
        content_mean, content_std = calc_mean_std(content_feat)
        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def split_and_combine_image(image, patch_index, patch_size=14, grid_size=37):
    image = np.array(image)
    h, w, c = image.shape[0], image.shape[1], image.shape[2]
    ph, pw = h // grid_size, w // grid_size
    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            patch = image[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :]
            patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LANCZOS4)
            patches.append(patch)
    patches = np.array(patches)
    patches = patches[patch_index]
    new_image = np.zeros((ph*grid_size, pw*grid_size, c), dtype=np.uint8)
    for idx, patch in enumerate(patches):
        i = idx // grid_size
        j = idx % grid_size
        new_image[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :] = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LANCZOS4)
    new_image = Image.fromarray(new_image)
    return new_image


def shuffle_image_patches(img, w=7, h=7):
    img_width, img_height = img.size

    patch_width = img_width // w
    patch_height = img_height // h
    
    new_width = patch_width * w
    new_height = patch_height * h
    img = img.resize((new_width, new_height))
    
    patches = []
    for i in range(h):
        for j in range(w):
            left = j * patch_width
            upper = i * patch_height
            right = left + patch_width
            lower = upper + patch_height
            patch = img.crop((left, upper, right, lower))
            patches.append(patch)
    
    random.shuffle(patches)
    
    new_img = Image.new('RGB', (new_width, new_height))
    for idx, patch in enumerate(patches):
        row = idx // w
        col = idx % w
        x = col * patch_width
        y = row * patch_height
        new_img.paste(patch, (x, y))
    return new_img


def tile_and_resize_image(img, h=7, w=7):
    orig_width, orig_height = img.size
    
    tiled_width = orig_width * w
    tiled_height = orig_height * h
    tiled_img = Image.new('RGB', (tiled_width, tiled_height))
    
    for i in range(h):
        for j in range(w):
            tiled_img.paste(img, (j * orig_width, i * orig_height))
    
    resized_img = tiled_img.resize((orig_width, orig_height), Image.LANCZOS)
    return resized_img