# XADE — Thesis References

All academic references behind the methods in XADE, organised by where they
appear in the system. Use this as the source-of-truth for the thesis
bibliography and for defending citations during the defence.

> **Important — verify before submitting.** Authors, years, and venues below
> are pulled from working knowledge. Click each DOI / arXiv link and confirm
> the entry against the canonical record before pasting into your reference
> manager. Use Zotero / Mendeley to fetch BibTeX from the DOI rather than
> typing entries by hand.

---

## 1. Papers formally cited in the codebase

These four papers have inline citations in our source code docstrings.

| Paper | What we use | Code location |
|---|---|---|
| **Kamali et al. (2024).** *How to Distinguish AI-Generated Images from Authentic Photographs.* IEEE Access, 12, 78823–78843. DOI: [10.1109/ACCESS.2024.3409217](https://doi.org/10.1109/ACCESS.2024.3409217) | The 6-category face taxonomy (anatomical + stylistic). | [`categories.py:32-37`](../backend/app/services/categories.py#L32-L37) |
| **Yu et al. (2018).** *BiSeNet: Bilateral Segmentation Network for Real-Time Semantic Segmentation.* ECCV. arXiv: [1808.00897](https://arxiv.org/abs/1808.00897) | The face-parsing model (per-pixel masks for 19 face classes). | [`face_parser.py:22-26`](../backend/app/services/face_parser.py#L22-L26) |
| **Lee et al. (2020).** *MaskGAN: Towards Diverse and Interactive Facial Image Manipulation.* CVPR. arXiv: [1907.11922](https://arxiv.org/abs/1907.11922) | The CelebAMask-HQ dataset (defines the 19 face classes BiSeNet was trained on). | [`face_parser.py:25`](../backend/app/services/face_parser.py#L25) |
| **Jiang et al. (2021).** *LayerCAM: Exploring Hierarchical Class Activation Maps for Localization.* IEEE TIP, 30, 5875–5888. DOI: [10.1109/TIP.2021.3089943](https://doi.org/10.1109/TIP.2021.3089943) | The CAM method we switched to from Grad-CAM. | [`gradcam_service.py:51-55`](../backend/app/services/gradcam_service.py#L51-L55) |

---

## 2. Forensic-method papers (must add to thesis bibliography)

These back the three pixel-level signals (sharpness, HF energy, ELA). They
are **not** currently cited inline in the codebase — they need to go in the
thesis methodology section so the metric choices are defensible.

| Paper | Backs |
|---|---|
| **Pech-Pacheco et al. (2000).** *Diatom Autofocusing in Brightfield Microscopy: A Comparative Study.* ICPR, vol. 3, 314–317. DOI: [10.1109/ICPR.2000.903548](https://doi.org/10.1109/ICPR.2000.903548) | **Laplacian variance** as a sharpness/focus measure. Originally an autofocus metric, repurposed in image-quality assessment and forensics. |
| **Wang et al. (2020).** *CNN-Generated Images Are Surprisingly Easy to Spot… for Now.* CVPR. arXiv: [1912.11035](https://arxiv.org/abs/1912.11035) | **Spectral-domain detectability of GAN-generated images.** Foundational paper showing synthesised images leave systematic frequency-domain signatures. |
| **Frank et al. (2020).** *Leveraging Frequency Analysis for Deep Fake Image Recognition.* ICML. arXiv: [2003.08685](https://arxiv.org/abs/2003.08685) | **High-frequency energy as a deepfake-detection signal.** Direct backing for our outer-spectrum analysis. |
| **Durall et al. (2020).** *Watch Your Up-Convolution: CNN-Based Generative Deep Neural Networks Are Failing to Reproduce Spectral Distributions.* CVPR. arXiv: [1911.00686](https://arxiv.org/abs/1911.00686) | **Why** GANs / diffusion models suppress high frequencies — the upsampling mechanism behind the HF deficit. |
| **Krawetz, N. (2007).** *A Picture's Worth: Digital Image Analysis and Forensics.* Black Hat Briefings USA. White paper: [hackerfactor.com](http://www.hackerfactor.com/papers/bh-usa-07-krawetz-wp.pdf) | **Error Level Analysis (ELA).** Original introduction. Industry paper — see alternatives below for academic substitutes. |

### Academic alternatives for the ELA reference

If your supervisor wants a peer-reviewed citation for ELA instead of Krawetz:

| Paper | Why use it |
|---|---|
| **Farid, H. (2009).** *Image Forgery Detection: A Survey.* IEEE Signal Processing Magazine, 26(2), 16–25. DOI: [10.1109/MSP.2008.931079](https://doi.org/10.1109/MSP.2008.931079) | Reviews ELA among other forgery-detection techniques in a peer-reviewed venue. |
| **Mahdian, B., & Saic, S. (2010).** *A Bibliography on Blind Methods for Identifying Image Forgery.* Signal Processing: Image Communication, 25(6), 389–399. DOI: [10.1016/j.image.2010.05.003](https://doi.org/10.1016/j.image.2010.05.003) | Treats ELA as part of a survey of blind forgery detection. |

---

## 3. Architecture and dataset citations

Things you depend on but aren't equation sources — still need to be cited.

| Paper | What for |
|---|---|
| **Tan, M., & Le, Q. (2019).** *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML. arXiv: [1905.11946](https://arxiv.org/abs/1905.11946) | **The detector architecture (EfficientNet-B4).** Critical citation. |
| **Selvaraju et al. (2017).** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV. arXiv: [1610.02391](https://arxiv.org/abs/1610.02391) | **The CAM baseline you ablate against.** Essential for the "why LayerCAM not Grad-CAM" justification. |
| **Karras et al. (2018).** *Progressive Growing of GANs for Improved Quality, Stability, and Variation.* ICLR. arXiv: [1710.10196](https://arxiv.org/abs/1710.10196) | **The FFHQ dataset** (training real images + study real images). |
| **Karras et al. (2019).** *A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN).* CVPR. arXiv: [1812.04948](https://arxiv.org/abs/1812.04948) | The StyleGAN family origin paper. |
| **Karras et al. (2021).** *Alias-Free Generative Adversarial Networks (StyleGAN3).* NeurIPS. arXiv: [2106.12423](https://arxiv.org/abs/2106.12423) | **StyleGAN3 specifically** — what generates the fake half of your dataset. Critical citation. |

---

## 4. Tools and libraries (cite if your venue requires)

| Tool | Citation |
|---|---|
| **PyTorch** | Paszke et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS. |
| **MediaPipe Face Landmarker** | Lugaresi, C. et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines.* arXiv: [1906.08172](https://arxiv.org/abs/1906.08172) |
| **pytorch-grad-cam library** | Gildenblat, J. (2021). *PyTorch library for CAM methods.* GitHub: [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) |
| **facexlib (BiSeNet wrapper)** | Wang, X. (2021). *facexlib: A collection of utility functions for face-related operations.* GitHub: [xinntao/facexlib](https://github.com/xinntao/facexlib) |

---

## 5. BibTeX block (paste into your `.bib` file, then verify)

```bibtex
% ── Cited in code ────────────────────────────────────────────────────────
@article{kamali2024distinguish,
  author  = {Kamali, Sepideh and Momeny, Mohammad and Rabbani, Hossein and Akbarizadeh, Gholamreza},
  title   = {How to Distinguish AI-Generated Images from Authentic Photographs},
  journal = {IEEE Access},
  volume  = {12},
  pages   = {78823--78843},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.3409217}
}

@inproceedings{yu2018bisenet,
  author    = {Yu, Changqian and Wang, Jingbo and Peng, Chao and Gao, Changxin and Yu, Gang and Sang, Nong},
  title     = {{BiSeNet}: Bilateral Segmentation Network for Real-Time Semantic Segmentation},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2018},
  eprint    = {1808.00897},
  archivePrefix = {arXiv}
}

@inproceedings{lee2020maskgan,
  author    = {Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
  title     = {{MaskGAN}: Towards Diverse and Interactive Facial Image Manipulation},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020},
  eprint    = {1907.11922},
  archivePrefix = {arXiv}
}

@article{jiang2021layercam,
  author  = {Jiang, Peng-Tao and Zhang, Chang-Bin and Hou, Qibin and Cheng, Ming-Ming and Wei, Yunchao},
  title   = {{LayerCAM}: Exploring Hierarchical Class Activation Maps for Localization},
  journal = {IEEE Transactions on Image Processing},
  volume  = {30},
  pages   = {5875--5888},
  year    = {2021},
  doi     = {10.1109/TIP.2021.3089943}
}

% ── Forensic-method backing ──────────────────────────────────────────────
@inproceedings{pechpacheco2000diatom,
  author    = {Pech-Pacheco, Jose Luis and Crist{\'o}bal, Gabriel and Chamorro-Martinez, Jesus and Fern{\'a}ndez-Valdivia, Joaquin},
  title     = {Diatom Autofocusing in Brightfield Microscopy: A Comparative Study},
  booktitle = {15th International Conference on Pattern Recognition (ICPR)},
  volume    = {3},
  pages     = {314--317},
  year      = {2000},
  doi       = {10.1109/ICPR.2000.903548}
}

@inproceedings{wang2020cnn,
  author    = {Wang, Sheng-Yu and Wang, Oliver and Zhang, Richard and Owens, Andrew and Efros, Alexei A.},
  title     = {{CNN}-Generated Images Are Surprisingly Easy to Spot... for Now},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020},
  eprint    = {1912.11035},
  archivePrefix = {arXiv}
}

@inproceedings{frank2020leveraging,
  author    = {Frank, Joel and Eisenhofer, Thorsten and Sch{\"o}nherr, Lea and Fischer, Asja and Kolossa, Dorothea and Holz, Thorsten},
  title     = {Leveraging Frequency Analysis for Deep Fake Image Recognition},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2020},
  eprint    = {2003.08685},
  archivePrefix = {arXiv}
}

@inproceedings{durall2020upconvolution,
  author    = {Durall, Ricard and Keuper, Margret and Keuper, Janis},
  title     = {Watch Your Up-Convolution: {CNN}-Based Generative Deep Neural Networks Are Failing to Reproduce Spectral Distributions},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020},
  eprint    = {1911.00686},
  archivePrefix = {arXiv}
}

@misc{krawetz2007picture,
  author = {Krawetz, Neal},
  title  = {A Picture's Worth: Digital Image Analysis and Forensics},
  year   = {2007},
  howpublished = {Black Hat Briefings USA},
  url    = {http://www.hackerfactor.com/papers/bh-usa-07-krawetz-wp.pdf}
}

@article{farid2009imageforgery,
  author  = {Farid, Hany},
  title   = {Image Forgery Detection: A Survey},
  journal = {IEEE Signal Processing Magazine},
  volume  = {26},
  number  = {2},
  pages   = {16--25},
  year    = {2009},
  doi     = {10.1109/MSP.2008.931079}
}

@article{mahdian2010bibliography,
  author  = {Mahdian, Babak and Saic, Stanislav},
  title   = {A Bibliography on Blind Methods for Identifying Image Forgery},
  journal = {Signal Processing: Image Communication},
  volume  = {25},
  number  = {6},
  pages   = {389--399},
  year    = {2010},
  doi     = {10.1016/j.image.2010.05.003}
}

% ── Architecture and datasets ────────────────────────────────────────────
@inproceedings{tan2019efficientnet,
  author    = {Tan, Mingxing and Le, Quoc V.},
  title     = {{EfficientNet}: Rethinking Model Scaling for Convolutional Neural Networks},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2019},
  eprint    = {1905.11946},
  archivePrefix = {arXiv}
}

@inproceedings{selvaraju2017gradcam,
  author    = {Selvaraju, Ramprasaath R. and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  title     = {{Grad-CAM}: Visual Explanations from Deep Networks via Gradient-Based Localization},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year      = {2017},
  eprint    = {1610.02391},
  archivePrefix = {arXiv}
}

@inproceedings{karras2018progressive,
  author    = {Karras, Tero and Aila, Timo and Laine, Samuli and Lehtinen, Jaakko},
  title     = {Progressive Growing of {GAN}s for Improved Quality, Stability, and Variation},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018},
  eprint    = {1710.10196},
  archivePrefix = {arXiv}
}

@inproceedings{karras2019stylegan,
  author    = {Karras, Tero and Laine, Samuli and Aila, Timo},
  title     = {A Style-Based Generator Architecture for Generative Adversarial Networks},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019},
  eprint    = {1812.04948},
  archivePrefix = {arXiv}
}

@inproceedings{karras2021stylegan3,
  author    = {Karras, Tero and Aittala, Miika and Laine, Samuli and H{\"a}rk{\"o}nen, Erik and Hellsten, Janne and Lehtinen, Jaakko and Aila, Timo},
  title     = {Alias-Free Generative Adversarial Networks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2021},
  eprint    = {2106.12423},
  archivePrefix = {arXiv}
}

% ── Tools / libraries ────────────────────────────────────────────────────
@inproceedings{paszke2019pytorch,
  author    = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and K{\"o}pf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
  title     = {{PyTorch}: An Imperative Style, High-Performance Deep Learning Library},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2019}
}

@misc{lugaresi2019mediapipe,
  author = {Lugaresi, Camillo and Tang, Jiuqiang and Nash, Hadon and McClanahan, Chris and Uboweja, Esha and Hays, Michael and Zhang, Fan and Chang, Chuo-Ling and Yong, Ming Guang and Lee, Juhyun and Chang, Wan-Teh and Hua, Wei and Georg, Manfred and Grundmann, Matthias},
  title  = {{MediaPipe}: A Framework for Building Perception Pipelines},
  year   = {2019},
  eprint = {1906.08172},
  archivePrefix = {arXiv}
}

@misc{gildenblat2021gradcam,
  author = {Gildenblat, Jacob and {contributors}},
  title  = {{PyTorch} library for {CAM} methods},
  year   = {2021},
  url    = {https://github.com/jacobgil/pytorch-grad-cam}
}

@misc{wang2021facexlib,
  author = {Wang, Xintao and {contributors}},
  title  = {facexlib: A Collection of Utility Functions for Face-Related Operations},
  year   = {2021},
  url    = {https://github.com/xinntao/facexlib}
}
```

---

## 6. Suggested next step

Open Zotero / Mendeley, search each DOI or arXiv ID, and import them into a
`XADE-thesis` collection. Cross-check any entry that looks odd against the
publisher's canonical record. Don't trust the BibTeX block above blindly — it's
a starting point, not a final bibliography.

If your supervisor asks for a citation list during the defence, you can refer
to this file or your reference manager.
