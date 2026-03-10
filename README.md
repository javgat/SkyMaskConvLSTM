<div align="center">

# [Solar Energy 2026] SkyMaskConvLSTM

Multi-frame cloud prediction in all-sky images from RGB images and segmented masks

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
    as the results indicated a bad performance. Its code and implementation may be useful.
  - `ConvLSTM`: Software related to the main networks & models: RGBConvLSTM and MaskConvLSTM. The evaluation and
    comparison of their performance is mainly carried out in `codes/ConvLSTM/Comparison.ipynb`.
  - `segmenter`: Software related to the evaluation of the ancillary GOA-UVa All-Sky Segmentation U-Net Model used.
- `docs`: Plots, diagrams and other figures.

## Acknowledgments

* [github.com/ndrplz/ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch)
* [github.com/yuhao-nie/SkyGPT](https://github.com/yuhao-nie/SkyGPT)
* [doi.org/10.5281/zenodo.18894938](https://doi.org/10.5281/zenodo.18894938)
