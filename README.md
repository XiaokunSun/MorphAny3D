# MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing

[**Project Page**](https://xiaokunsun.github.io/MorphAny3D.github.io) | [**Arxiv**](https://arxiv.org/pdf/2408.09126)

Official repo of "MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing“

[Xiaokun Sun](https://xiaokunsun.github.io), [Zeyu Cai](https://zcai0612.github.io), [Hao Tang](https://ha0tang.github.io), [Ying Tai](https://tyshiwo.github.io/index.html),  [Jian Yang](https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ), [Zhenyu Zhang](https://jessezhang92.github.io)


<p align="center"> All Code will be released soon... 🚀🚀🚀 </p>

Abstract: 3D morphing remains challenging due to the difficulty of generating semantically consistent and temporally smooth deformations, especially across categories. We present MorphAny3D, a training-free framework that leverages Structured Latent (SLAT) representations for high-quality 3D morphing. Our key insight is that intelligently blending source and target SLAT features within the attention mechanisms of 3D generators naturally produces plausible morphing sequences. To this end, we introduce Morphing Cross-Attention (MCA), which fuses source and target information for structural coherence, and Temporal-Fused Self-Attention (TFSA), which enhances temporal consistency by incorporating features from preceding frames. An orientation correction strategy further mitigates the pose ambiguity within the morphing steps. Extensive experiments show that our method generates state-of-the-art morphing sequences, even for challenging cross-category cases. MorphAny3D further supports advanced applications such as decoupled morphing and 3D style transfer, and can be generalized to other SLAT-based generative models.

<p align="center">
    <img src="assets/Teaser.png">
</p>

## BibTeX

```bibtex
@article{sun2024barbie,
  title={Barbie: Text to Barbie-Style 3D Avatars},
  author={Sun, Xiaokun and Zhang, Zhenyu and Tai, Ying and Tang, Hao and Yi, Zili and Yang, Jian},
  journal={arXiv preprint arXiv:2408.09126},
  year={2024}
}
```
