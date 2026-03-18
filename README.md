<div align="center">

# [Solar Energy 2026] SkyMaskConvLSTM

Multi-frame cloud prediction in all-sky images from RGB images and segmented masks

[![Paper DOI: 10.1016/j.solener.2026.114515](https://img.shields.io/badge/paper_doi-10.1016%2Fj.solener.2026.114515-informational?style=for-the-badge)](https://doi.org/10.1016/j.solener.2026.114515)

![graphical-abstract](./docs/other/grabs.png)

</div>

## Abstract

This paper presents a comparative study on the impact of input representation on
deterministic artificial intelligence models for short-term multi-frame prediction in all-sky
images. This work compares a model operating on 8-bit RGB all-sky images with a
model that shares the same backbone, but operating directly on semantically
segmented masks that encode cloud-related classes. Using an available sky
segmentation model, predictions are evaluated in the segmentation label space using
segmenter-derived masks as a proxy reference. Within this evaluation framework, the
use of semantic masks as input for short-term prediction leads to improved temporal
stability and higher agreement across standard segmentation metrics such as
intersection over union, Dice coefficient, and categorical cross-entropy. While these
results suggest potential relevance for weather and solar energy nowcasting
applications, further validation against physical irradiance measurements is required.

## Repository Structure

- `codes`: Software for the implementation, training and evaluation of the models.
  - `common`: Functions and scripts useful for different networks.
  - `CNNLSTM`: Software related to the CNN-LSTM models. They were not included in the journal article,
    due to poor performance, but the code may still be of interest.
  - `ConvLSTM`: Software related to the main networks & models: RGBConvLSTM and MaskConvLSTM. The evaluation and
    comparison of their performance is mainly carried out in `codes/ConvLSTM/Comparison.ipynb`.
  - `segmenter`: Software related to the evaluation of the ancillary GOA-UVa all-sky segmentation U-Net model.
- `docs`: Plots, diagrams and additional figures.

## Citation

If you find our study useful to your research, please cite with:
```
@article{gaton2026skymaskconvlstm,
  title = {Multi-frame cloud prediction in all-sky images from RGB images and segmented masks},
  author = {Javier Gatón and Roberto Román and Cesar Guzman and Daniel González-Fernández and Bruno Longarela and Carlos Toledano and Ramiro González},
  journal = {Solar Energy},
  volume = {311},
  pages = {114515},
  year = {2026},
  issn = {0038-092X},
  publisher = {Elsevier},
  doi = {10.1016/j.solener.2026.114515},
  url = {https://www.sciencedirect.com/science/article/pii/S0038092X26002033},
}
```

## Acknowledgments

* [github.com/ndrplz/ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch)
* [github.com/yuhao-nie/SkyGPT](https://github.com/yuhao-nie/SkyGPT)
* [doi.org/10.5281/zenodo.18894938](https://doi.org/10.5281/zenodo.18894938)
