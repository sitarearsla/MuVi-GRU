# MuVi-GRU Implementation

A PyTorch implementation of a simple baseline for arousal and valence estimation for audio. 
You can check the original Tensorflow implementation written by [Phoebe Chua et al.](https://github.com/amaai-lab/muvi).
Some codes for data processing are brought from the original version, thanks to the authors.

<!-- ![demo](./img/demo.jpg) -->

This is the code for the paper

```
@article{chua2022predicting,
  title={Predicting emotion from music videos: exploring the relative contribution of visual and auditory information to affective responses},
  author={Chua, Phoebe and Makris, Dimos and Herremans, Dorien and Roig, Gemma and Agres, Kat},
  journal={arXiv preprint arXiv:2202.10453},
  year={2022}
}
```

## WIP


 - [x] Training code
 - [x] Testing code

### Dataset Files

 - [x] video_urls.csv: Contains the YouTube ids of MuVi dataset.
 - [x] av_data.csv: Includes the dynamic (continuous) annotations for Valence and Arousal
 - [x] emobase_features: Includes include the extracted audio for the videos in MuVi dataset

## Dependencies

* sklearn
* numpy
* pandas
* torch
* scipy
* audtorch
* matplotlib

## Installation

1. First, clone this repository:
    ```
    git clone --recursive https://github.com/sitarearsla/MuVi-GRU
    ```
2. Download the pre-extracted emobase_features, video_urls.csv and av_data.csv from the [original implementation](https://github.com/amaai-lab/muvi) 
    ```
    unzip combined_visual_features.rar
    rm combined_visual_features.rar
    ```

## Usage

### Data preprocess

### Train

1. Train on emobase_features:

    train the model:
    ```
    python train_model.py 
    ```

    You will get the training and validation loss curves like:

    ![alt text](https://github.com/sitarearsla/MuVi-GRU/blob/main/gru_loss.png)

