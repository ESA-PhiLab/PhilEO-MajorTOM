# PhilEO-MajorTOM: Scaling-up the pretraining of Geospatial Foundation Models

![Banner](images/esa.png)
![Banner](images/leonardo.png)
![Banner](images/leonardo2.png)

# Table Of Contents
1. [Introduction](#introduction)
2. [Datasets](#data)
3. [New Models](#newmodels)
4. [Usage](#usage)
5. [Models and Data](#models)


## Introduction <a name="introduction"></a>
This repository is an extension of the previously introduced [PhilEO Bench](http://arxiv.org/pdf/2401.04464), and is linked to [Paper](http://arxiv.org/pdf/2506.14765). The latter paper is a product of a collaboration between ESA's Phi-lab and Leonardo Labs. Due to protocol constraints, we have open-sourced a selection of files.   
  
The PhilEO Bench serves as a framework that allows users to benchmark various
Geospatial Foundation Models (GFMs) against each other on three downstream tasks: road density estimation, building density estimation, and land cover classification. The first two tasks are pixel-wise regression, while the last downstream task is semantic segmentation using 11 classes.   

In [Paper](http://arxiv.org/pdf/2506.14765), we expand on PhilEO Bench, by scaling up the 
pretraining of the Geo-Aware U-Net to subsets extracted from [MajorTOM](https://github.com/ESA-PhiLab/Major-TOM). Moreover, we demonstrate that the PhilEO ViT UPerNet outperforms its C-NN decoder-based counterparts across all three downstream tasks. This repo can be considered a cleaned-up version of the
previously mentioned PhilEO Bench repo, with additional files related to pretraining and fine-tuning the aforemetioned models.


## Datasets <a name="data"></a>
The datasets used for pretraining are extracted from the MajorTOM repo. In particular, we pretrained the Geo-Aware U-Net on the MajorTOM 23TB Sentinel-2 dataset, and its smaller 2TB subset, called FastTOM.
This yields increased performance w.r.t. the previously used 0.5TB PhilEO Globe dataset. For fine-tuning, we use the labelled 0.4TB PhilEO Bench [downstream](https://huggingface.co/datasets/PhilEO-community/PhilEO-downstream) dataset.

The file ```majortom.py```, found in the ```data``` folder, contains a PyTorch implementation for formatting the extracted data from MajorTOM.


## New Models <a name="newmodels"></a>
In addition to the already published models from the PhilEO Bench, which can be found in the folder ```phileo-bench```, we also added the following files to the aforementioned folder:

- ```decoder_UperNet.py```: contains the standard [UPerNet](https://arxiv.org/abs/1807.10221) implementation.

- ```model_PhiViTUperNet.py```: contains the implementation for the PhilEO ViT UPerNet.

The folder ```model``` holds 2 model files:

- ```phileo_cnn.py```: the [GeoDINO](https://meetingorganizer.copernicus.org/EGU25/EGU25-18029.html) architecture based on a U-Net design (i.e. the PhilEO C-NN).

- ```phileo_vit.py```: an adaptation to the GeoDINO architecture, using a ViT instead of a U-Net (i.e. the PhilEO ViT)


## Usage <a name="usage"></a>
This repo offers a better use of computational resources, by leveraging the power of Distributed Data Parallel training in PyTorch, and thus effectively allowing you to utilize all your available GPUs. In particular, the following scripts can be used for fine-tuning, using a DDP paradigm: 

- ```train_model_ddp.py```: fine-tune the PhilEO C-NN. 

- ```train_model_vit_ddp.py```: fine-tune the PhilEO ViT. 

## Models and Data <a name="models"></a>

Model weights: [Models](http://huggingface.co/NikolaosDionelis2023/GeoAware-ModelWeights/tree/main)   

Data: [Data S-2](http://huggingface.co/datasets/NikolaosDionelis2023/s2-phileobench/tree/main) 

Also: Data: [Data S-1](http://huggingface.co/datasets/NikolaosDionelis2023/s1-phileobench/tree/main) 

Paper: [PhilEO MajorTOM](http://arxiv.org/pdf/2506.14765)     

Also: Paper: [PhilEO Scaling-Up](http://arxiv.org/pdf/2506.14765v1)    

----------------

Also: PhilEO Bench: [IGARSS Paper](http://phileo-bench.github.io/)   

GitHub: [Code](http://github.com/ESA-PhiLab/PhilEO-Bench)    

[PhilEO Bench](http://arxiv.org/pdf/2401.04464)  

----------------

## Usage

To run the model:  

git clone https://github.com/ESA-PhiLab/PhilEO-MajorTOM.git    

cd PhilEO-MajorTOM   

(pip install -r requirements.txt)

python train_model_ddp.py 

(or python train_model_vit_ddp.py)




