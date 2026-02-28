<div align="center">

# [CVPR 2026] MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing

<div>
    <a href="https://xiaokunsun.github.io"><strong>Xiaokun Sun</strong></a><sup>1</sup>,
    <a href="https://zcai0612.github.io"><strong>Zeyu Cai</strong></a><sup>1</sup>,
    <a href="https://ha0tang.github.io"><strong>Hao Tang</strong></a><sup>2</sup>,
    <a href="https://tyshiwo.github.io/index.html"><strong>Ying Tai</strong></a><sup>1</sup>,
    <a href="https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ"><strong>Jian Yang</strong></a><sup>1</sup>,
    <a href="https://jessezhang92.github.io"><strong>Zhenyu Zhang</strong></a><sup>1*</sup>
</div>

<div>
    <sup>1</sup><strong>Nanjing University</strong> &nbsp;&nbsp;
    <sup>2</sup><strong>Peking University</strong>
</div>

<div>
    <sup>*</sup><strong>Corresponding Author</strong>
</div>

<br>

[![ArXiv](https://img.shields.io/badge/ArXiv-2601.00204-b31b1b.svg)](https://arxiv.org/pdf/2601.00204)
[![Project Page](https://img.shields.io/badge/Project%20Page-MorphAny3D-Green.svg)](https://xiaokunsun.github.io/MorphAny3D.github.io)

<br>
<img src="assets/teaser.png" alt="Teaser" width="100%">
</div>

## 🔨 Installation
Tested on **Ubuntu 20.04**, **Python 3.10**, **NVIDIA A6000**, **CUDA 11.8**, and **PyTorch 2.4.0**. Follow the steps below to set up the environment.

1. Clone the repo:
    ```bash
    git clone https://github.com/XiaokunSun/MorphAny3D.git
    cd MorphAny3D
    ```

2. Setup the environment:

    As MorphAny3D builds upon [TRELLIS](https://github.com/microsoft/TRELLIS). You can find more details about the dependencies in the TRELLIS repository.
    ```bash
    bash ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
    conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu118_pyt240.tar.bz2 # Note: Please ensure the pytorch3d version matches your Python, CUDA and Torch versions
    ```

3. Download pretrained models:

    We do not modify the pretrained models of TRELLIS. The weights will be automatically downloaded when you run:
    ```sh
    TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    ```
    Optionally, you can manually download the weights from [HuggingFace](https://huggingface.co/microsoft/TRELLIS-image-large) and change the path in the above command to the local path.
    ```sh
    TrellisImageTo3DPipeline.from_pretrained("path/to/local/directory")
    ``` 

## 🕺 Inference
```bash
# 3D Morphing
python ./example_3Dmorphing.py
```

## 🪄 Application
```bash
# Disentangled 3D Morphing
python ./example_disentangled_3Dmorphing.py
# Dual-Target 3D Morphing
python ./example_dual_target_3Dmorphing.py
# 3D Style Transfer
python ./example_3Dstyle_transfer.py
```

## 💖 Acknowledgements
This code builds upon [TRELLIS](https://github.com/microsoft/TRELLIS).
We sincerely thank the authors for their great work and open-sourcing the code.

## 📚 Citation
If you find our work helpful for your research, please consider starring this repository ⭐ and citing our work:
```bibtex
@inproceedings{sun2026morphany3d,
  title={MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing},
  author={Sun, Xiaokun and Cai, Zeyu and and Tang, Hao and Tai, Ying and Yang, Jian and Zhang, Zhenyu},
  booktitle={CVPR},
  year={2026}
}
```