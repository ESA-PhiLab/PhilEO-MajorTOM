# # Use: To run model: python n_shot_experiment3.py --read_yaml=default_args3.yml  

from utils_mamba2.pos_embed import *

import os
import torch
from functools import partial
from torchinfo import summary
import torch.nn as nn
from datetime import date
import argparse 
import sys; sys.path.append("../")     

from models.model_Baseline import BaselineNet
from models.model_CoreCNN_versions import CoreUnet_nano, CoreUnet_tiny, CoreUnet_base, CoreUnet_large, CoreUnet_huge, Core_nano
from models.model_Mixer_versions import Mixer_nano, Mixer_tiny, Mixer_base, Mixer_large, Mixer_huge
from models.model_LinearViT_versions import LinearViT_base, LinearViT_large, LinearViT_huge
from models.model_AutoEncoderViT_versions import AutoencoderViT_base, AutoencoderViT_large, AutoencoderViT_huge
from models.model_GeoAwarePretrained import MixerGeoPretrained, get_mixer_kwargs, get_core_encoder_kwargs, CoreEncoderGeoPretrained, CoreEncoderGeoPretrained_combined, CoreEncoderGeoAutoEncoder
from models.model_GeoAwarePretrained_classifier import CoreEncoderGeoPretrained_Classifier
from models.model_AutoEncoderViTPretrained import vit_cnn, vit_cnn_gc, vit_large, get_core_decoder_kwargs
from models.model_AutoEncoderViTPretrained_wSkip import vit_cnn_wSkip, vit_cnn_gc_wSkip, vit_large_wSkip
from models.model_AutoEncoderViTPretrained_classifier import vit_cnn_classifier, vit_cnn_gc_classifier
from models.model_CoreVAE import CoreVAE_nano
from models.model_SatMAE import satmae_vit_cnn
from models.models_Prithvi import prithvi
from models.model_Seco import seasonal_contrast
from models.model_Resnet50 import resnet

from utils import data_protocol
from utils import load_data
import math    
import torch.nn.functional as F       

#from utils import training_loops          
from utils import training_loops2 as training_loops

from utils.training_utils import read_yaml    
torch.manual_seed(123456) 
CNN_LIST = ['baseline_cnn', 'core_unet_nano','core_unet_tiny','core_unet_base', 'core_unet_large', 'core_unet_huge',
            'core_vae_nano', 'resnet_imagenet', 'resnet', 'core_encoder_nano', 'resnet_imagenet_classifier']

VIT_CNN_LIST = ['vit_cnn_base', 'vit_cnn_base_wSkip'] 

MIXER_LIST = ['mixer_nano', 'mixer_tiny', 'mixer_base', 'mixer_large', 'mixer_huge']

VIT_LIST = ['linear_vit_base', 'linear_vit_larger', 'linear_vit_huge',
            'autoencoder_vit_base', 'autoencoder_vit_large', 'autoencoder_vit_huge']

CNN_PRETRAINED_LIST = ['GeoAware_core_nano', 'GeoAware_core_tiny', 'GeoAware_mixer_nano', 'GeoAware_mixer_tiny',
                       'GeoAware_contrastive_core_nano', 'GeoAware_mh_pred_core_nano', 'GeoAware_combined_core_nano',
                       'GeoAware_core_autoencoder_nano', 'seasonal_contrast',
                       'GeoAware_core_nano_classifier', 'GeoAware_contrastive_core_nano_classifier',
                       'GeoAware_mh_pred_core_nano_classifier', 'seasonal_contrast_classifier'
                       ]

VIT_CNN_PRETRAINED_LIST = ['prithvi', 'vit_cnn', 'vit_cnn_gc', 'SatMAE', 'SatMAE_classifier', 'vit_cnn_gc_classifier',
                           'vit_cnn_classifier', 'prithvi_classifier', 'vit_cnn_wSkip', 'vit_cnn_gc_wSkip']

MODELS_224 = ['seasonal_contrast', 'resnet_imagenet', 'resnet', 'seasonal_contrast_classifier', 'resnet_imagenet_classifier']
MODELS_224_r30 = ['prithvi', 'prithvi_classifier']

MODEL_LIST = CNN_LIST + MIXER_LIST + VIT_LIST + CNN_PRETRAINED_LIST + VIT_CNN_LIST + VIT_CNN_PRETRAINED_LIST
DOWNSTREAM_LIST = ['lc', 'building', 'roads', 'lc_classification', 'building_classification', 'roads_classification']

def get_trainer(model_name, downstream_task, epochs, lr, model, device, lr_scheduler, warmup, early_stop, dl_train,
                dl_val, dl_test, NAME, OUTPUT_FOLDER, vis_val, warmup_steps, warmup_gamma):

    if model_name in (CNN_LIST + MIXER_LIST + VIT_CNN_LIST + CNN_PRETRAINED_LIST + VIT_CNN_PRETRAINED_LIST):
        if downstream_task == 'roads' or downstream_task == 'building':
            trainer = training_loops.TrainBase(epochs=epochs, lr=lr, model=model, device=device,
                                               lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                               train_loader=dl_train,
                                               val_loader=dl_val, test_loader=dl_test, name=NAME,
                                               out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                               warmup_steps=warmup_steps, warmup_gamma=warmup_gamma)
        elif downstream_task == 'lc':
            trainer = training_loops.TrainLandCover(epochs=epochs, lr=lr, model=model, device=device,
                                                    lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                                    train_loader=dl_train,
                                                    val_loader=dl_val, test_loader=dl_test, name=NAME,
                                                    out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                                    warmup_steps=warmup_steps, warmup_gamma=warmup_gamma)
        elif downstream_task == 'building_classification':
            trainer = training_loops.TrainClassificationBuildings(epochs=epochs, lr=lr, model=model, device=device,
                                                                  lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                                                  train_loader=dl_train,
                                                                  val_loader=dl_val, test_loader=dl_test, name=NAME,
                                                                  out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                                                  warmup_steps=warmup_steps, warmup_gamma=warmup_gamma
                                                                  )

        elif downstream_task == 'lc_classification':
            trainer = training_loops.TrainClassificationLC(epochs=epochs, lr=lr, model=model, device=device,
                                                           lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                                           train_loader=dl_train,
                                                           val_loader=dl_val, test_loader=dl_test, name=NAME,
                                                           out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                                           warmup_steps=warmup_steps, warmup_gamma=warmup_gamma)

        elif downstream_task == 'roads_classification':
            trainer = training_loops.TrainClassificationRoads(epochs=epochs, lr=lr, model=model, device=device,
                                                           lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                                           train_loader=dl_train,
                                                           val_loader=dl_val, test_loader=dl_test, name=NAME,
                                                           out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                                           warmup_steps=warmup_steps, warmup_gamma=warmup_gamma)

    elif model_name in (VIT_LIST):
        if downstream_task == 'roads' or downstream_task == 'building':
            trainer = training_loops.TrainViT(epochs=epochs, lr=lr, model=model, device=device,
                                              lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop, train_loader=dl_train,
                                              val_loader=dl_val, test_loader=dl_test, name=NAME,
                                              out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                              warmup_steps=warmup_steps, warmup_gamma=warmup_gamma)

        elif downstream_task == 'lc':
            trainer = training_loops.TrainViTLandCover(epochs=epochs, lr=lr, model=model, device=device,
                                                       lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                                       train_loader=dl_train,
                                                       val_loader=dl_val, test_loader=dl_test, name=NAME,
                                                       out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                                       warmup_steps=warmup_steps, warmup_gamma=warmup_gamma)

    if model_name == 'core_vae_nano':
        trainer = training_loops.TrainVAE(epochs=epochs, lr=lr, model=model, device=device,
                                          lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                          train_loader=dl_train,
                                          val_loader=dl_val, test_loader=dl_test, name=NAME,
                                          out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                          warmup_steps=warmup_steps, warmup_gamma=warmup_gamma)

    return trainer

def get_models(model_name, input_channels, output_channels, input_size):
    if model_name == 'baseline_cnn':
        return BaselineNet(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_nano':
        return CoreUnet_nano(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_encoder_nano':
        return Core_nano(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_tiny':
        return CoreUnet_tiny(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_base':
        return CoreUnet_base(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_large':
        return CoreUnet_large(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_huge':
        return CoreUnet_huge(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'mixer_nano':
        return Mixer_nano(chw=(input_channels, input_size, input_size),
                          output_dim=output_channels)
    elif model_name == 'mixer_tiny':
        return Mixer_tiny(chw=(input_channels, input_size, input_size),
                          output_dim=output_channels)
    elif model_name == 'mixer_base':
        return Mixer_base(chw=(input_channels, input_size, input_size),
                          output_dim=output_channels)
    elif model_name == 'mixer_large':
        return Mixer_large(chw=(input_channels, input_size, input_size),
                           output_dim=output_channels)
    elif model_name == 'mixer_huge':
        return Mixer_huge(chw=(input_channels, input_size, input_size),
                          output_dim=output_channels)
    elif model_name == 'linear_vit_base':
        return LinearViT_base(chw=(input_channels, input_size, input_size),
                              output_dim=output_channels)
    elif model_name == 'linear_vit_large':
        return LinearViT_large(chw=(input_channels, input_size, input_size),
                               output_dim=output_channels)
    elif model_name == 'linear_vit_huge':
        return LinearViT_huge(chw=(input_channels, input_size, input_size),
                              output_dim=output_channels)
    elif model_name == 'autoencoder_vit_base':
        return AutoencoderViT_base(chw=(input_channels, input_size, input_size),
                                   output_dim=output_channels)
    elif model_name == 'autoencoder_vit_large':
        return AutoencoderViT_large(chw=(input_channels, input_size, input_size),
                                    output_dim=output_channels)
    elif model_name == 'autoencoder_vit_huge':
        return AutoencoderViT_huge(chw=(input_channels, input_size, input_size),
                                   output_dim=output_channels)
    elif model_name == 'core_vae_nano':
        return CoreVAE_nano(input_dim=input_channels, output_dim=10)

    elif model_name == 'vit_cnn_base':
        return vit_large(chw=(input_channels, input_size, input_size),
                         output_dim=output_channels)
    elif model_name == 'vit_cnn_base_wSkip':
        return vit_large_wSkip(chw=(input_channels, input_size, input_size),
                         output_dim=output_channels)
    elif model_name == 'resnet_imagenet':
        resnet_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return resnet(imagenet_weights=True, **resnet_kwargs)
    elif model_name == 'resnet_imagenet_classifier':
        resnet_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return resnet(imagenet_weights=True, classifier=True, **resnet_kwargs)
    elif model_name == 'resnet':
        resnet_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return resnet(imagenet_weights=False, **resnet_kwargs)

def get_models_pretrained(model_name, input_channels, output_channels, input_size, path_model_weights=None, freeze=False, device='cuda'):
    test_input = torch.rand((2,input_channels,input_size,input_size))

    if (model_name == 'GeoAware_core_nano' or model_name == 'GeoAware_contrastive_core_nano' or
            model_name == 'GeoAware_mh_pred_core_nano'):

        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano', full_unet=True)
        model = CoreEncoderGeoPretrained(output_channels, checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
        model(test_input)
        return model

    if (model_name == 'GeoAware_core_nano_classifier' or model_name == 'GeoAware_contrastive_core_nano_classifier' or
            model_name == 'GeoAware_mh_pred_core_nano_classifier'):

        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano', full_unet=False)
        model = CoreEncoderGeoPretrained_Classifier(checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
        model(test_input)
        return model

    if model_name == 'GeoAware_core_autoencoder_nano':
        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano', full_unet=True)
        model = CoreEncoderGeoAutoEncoder(output_channels, checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
        model(test_input)
        return model

    if model_name == 'GeoAware_combined_core_nano':
        sd_1 = torch.load(path_model_weights[0])
        sd_2 = torch.load(path_model_weights[1])
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano')
        model = CoreEncoderGeoPretrained_combined(output_channels, checkpoint_1=sd_1, checkpoint_2=sd_2,
                                                  core_encoder_kwargs=core_kwargs)
        model(test_input)
        return model
    
    if model_name == 'GeoAware_core_tiny':
        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_tiny', full_unet=True)
        model = CoreEncoderGeoPretrained(output_channels, checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
        model(test_input)
        return model
    
    if model_name == 'GeoAware_mixer_nano':   
        sd = torch.load(path_model_weights) 
        mixer_kwargs = get_mixer_kwargs(chw=(input_channels,input_size,input_size),output_dim=output_channels, mixer_size='mixer_nano')
        model =  MixerGeoPretrained(output_dim=output_channels, checkpoint=sd, mixer_kwargs=mixer_kwargs, freeze_body=freeze)
        model(test_input)
        return model 
    
    if model_name == 'GeoAware_mixer_tiny':
        sd = torch.load(path_model_weights)
        mixer_kwargs = get_mixer_kwargs(chw=(input_channels,input_size,input_size),output_dim=output_channels, mixer_size='mixer_tiny')
        model = MixerGeoPretrained(output_dim=output_channels, checkpoint=sd, mixer_kwargs=mixer_kwargs, freeze_body=freeze)
        model(test_input)
        return model 

    elif model_name == 'SatMAE':
        sd = torch.load(path_model_weights)
        satmae_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return satmae_vit_cnn(img_size=96, patch_size=8, in_chans=input_channels,
                              checkpoint=sd, freeze_body=freeze, classifier=False, **satmae_kwargs)

    elif model_name == 'SatMAE_classifier':
        sd = torch.load(path_model_weights)
        satmae_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return satmae_vit_cnn(img_size=96, patch_size=8, in_chans=input_channels,
                              checkpoint=sd, freeze_body=freeze, classifier=True, **satmae_kwargs)

    elif model_name == 'prithvi':
        sd = torch.load(path_model_weights, map_location=device)
        prithvi_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return prithvi(checkpoint=sd, freeze_body=freeze, **prithvi_kwargs)

    elif model_name == 'prithvi_classifier':
        sd = torch.load(path_model_weights, map_location=device)
        prithvi_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return prithvi(checkpoint=sd, freeze_body=freeze, classifier=True, **prithvi_kwargs)

    elif model_name == 'vit_cnn':
        sd = torch.load(path_model_weights, map_location=device)
        vit_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return vit_cnn(checkpoint=sd, freeze_body=freeze, **vit_kwargs)

    elif model_name == 'vit_cnn_wSkip':
        sd = torch.load(path_model_weights, map_location=device)
        vit_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return vit_cnn_wSkip(checkpoint=sd, freeze_body=freeze, **vit_kwargs)

    elif model_name == 'vit_cnn_classifier':
        sd = torch.load(path_model_weights, map_location=device)
        return vit_cnn_classifier(checkpoint=sd, freeze_body=freeze, output_dim=output_channels)

    elif model_name == 'vit_cnn_gc':
        sd = torch.load(path_model_weights, map_location=device)
        vit_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return vit_cnn_gc(checkpoint=sd, freeze_body=freeze, **vit_kwargs)

    elif model_name == 'vit_cnn_gc_wSkip':
        sd = torch.load(path_model_weights, map_location=device)
        vit_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return vit_cnn_gc_wSkip(checkpoint=sd, freeze_body=freeze, **vit_kwargs)

    elif model_name == 'vit_cnn_gc_classifier':
        sd = torch.load(path_model_weights, map_location=device)
        return vit_cnn_gc_classifier(checkpoint=sd, freeze_body=freeze, output_dim=output_channels)

    elif model_name == 'seasonal_contrast':
        seco_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return seasonal_contrast(checkpoint=path_model_weights, freeze_body=freeze,
                                 **seco_kwargs)

    elif model_name == 'seasonal_contrast_classifier':
        seco_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return seasonal_contrast(checkpoint=path_model_weights, freeze_body=freeze, classifier=True,
                                 **seco_kwargs)

def get_args():
    parser_yaml = argparse.ArgumentParser(description='Experiment TestBed for Phi-Leo Foundation Model Project')
    parser_yaml.add_argument('--read_yaml', type=str, help='take parameters from yaml path', default=None)

    parser = argparse.ArgumentParser(description='Experiment TestBed for Phi-Leo Foundation Model Project')
    parser.add_argument('--experiment_name', type=str, default=f'{date.today().strftime("%d%m%Y")}_experiment',
                        help='Experiment folder name')
    parser.add_argument('--model_name', type=str, choices=MODEL_LIST, required=True,
                        help='Select appropriate model')
    parser.add_argument('--lr', type=float, default=0.001, help='Set learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Set batch size')
    parser.add_argument('--epochs', type=int, default=250, help='Set training epochs')
    parser.add_argument('--early_stop', type=int, default=50, help='set training loop patience for early stopping')
    parser.add_argument('--lr_scheduler', type=str, default=None,
                        choices=[None, 'reduce_on_plateau', 'cosine_annealing'], help='select learning rate scheduler')
    parser.add_argument('--warmup', action="store_true", help='Enables linear 5 epoch warmup scheduler')
    parser.add_argument('--model_device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='select training device')
    parser.add_argument('--generator_device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='select training device')
    parser.add_argument('--num_workers', type=int, default=0, help='set number of workers')
    parser.add_argument('--vis_val', action="store_true", help='enable saving of intermediate visualization plots')
    parser.add_argument('--downstream_task', type=str, choices=DOWNSTREAM_LIST, required=True,
                        help='select downstream task')
    parser.add_argument('--input_channels', type=int, required=False, default=10, help='Define Number of input channels')
    parser.add_argument('--input_size', type=int, required=True, default=128, help='Define input size')
    parser.add_argument('--output_channels', type=int, required=True, default=1, help='Define Number of output channels')

    parser.add_argument('--regions', type=list, default=None, help='select regions to be included',
                        choices=[None, 'denmark-1', 'denmark-2', 'east-africa', 'egypt-1', 'eq-guinea', 'europe', 'ghana-1',
                                 'isreal-1', 'isreal-2', 'japan', 'nigeria', 'north-america', 'senegal', 'south-america',
                                 'tanzania-1', 'tanzania-2', 'tanzania-3', 'tanzania-4', 'tanzania-5', 'uganda-1'])
    parser.add_argument('--n_shot', type=int, default=None,
                        help='Loads n-samples of data from specified geographic regions')
    parser.add_argument('--split_ratio', type=float, default=None,
                        help='Loads a percentage of the data from specified geographic regions.')
    parser.add_argument('--augmentations', action="store_true", help='enables augmentations')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained weights')
    parser.add_argument('--freeze_pretrained', action="store_true", help='freeze pretrained model weights')
    parser.add_argument('--data_path_128_10m', type=str, default='/home/phimultigpu/phileo_NFS/phileo_data/downstream/downstream_dataset_patches_np/')
    parser.add_argument('--data_path_224_10m', type=str, default='/home/phimultigpu/phileo_NFS/phileo_data/downstream/downstream_dataset_patches_np_224/')
    parser.add_argument('--data_path_224_30m', type=str, default='/home/phimultigpu/phileo_NFS/phileo_data/downstream/downstream_dataset_patches_np_HLS/')
    parser.add_argument('--C', type=str, default='/home/phimultigpu/phileo_NFS/phileo_data/experiments')
    parser.add_argument('--data_parallel', type=bool, default=False)
    parser.add_argument('--device_ids', type=list, default=[0, 1, 2, 3])
    parser.add_argument('--warmp_steps', type=int, default=5)
    parser.add_argument('--warmup_gamma', type=int, default=10)

    return parser, parser_yaml   

def main(experiment_name:str, downstream_task:str, model_name:str, augmentations:bool=False, batch_size:int=16, 
         model_device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), generator_device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), num_workers:int=4,
         early_stop:int=25, epochs:int=250, input_channels:int=10, output_channels:int=1, input_size:int=128, lr:float=0.001, lr_scheduler:str=None,
         n_shot:int=None, split_ratio:float=0.1, regions:list=None,  vis_val:bool=True, warmup:bool=False , warmp_steps:int=5, warmup_gamma:int=10, pretrained_model_path:str=None, freeze_pretrained:bool=None,
         data_path_128_10m:str=None, data_path_224_10m:str=None, data_path_224_30m:str=None, output_path:str=None, data_parallel:bool=False, device_ids:list=None):
    """ main script for PhilEO Bench. Used to run model training experiments with randomly initialized and pre-trained models on a number of downstream tasks. 
        The script handles dataset creation (based on data protocol options selected), data preprocessing (based on downstream task & model type) & model, training, validation and testing. 

    Parameters
    ----------
        experiment_name (str): Experiment name 
        downstream_task (str): Select downstream task to test, validate and test on. Options: {DOWNSTREAM_LIST}
        model_name (str): Select model. Options:{MODEL_LIST}
        augmentations (bool, optional): Toggle on/off basic data augmentations (Rotation, Mirror, Noise). Defaults to False.
        batch_size (int, optional): Define training batch size. Defaults to 16.
        model_device (_type_, optional): Select model device. Defaults to torch.device('cuda' if torch.cuda.is_available() else 'cpu').
        generator_device (_type_, optional): Select dataloader device. Defaults to torch.device('cuda' if torch.cuda.is_available() else 'cpu').
        num_workers (int, optional): Select number of workers for dataloader. Defaults to 4.
        early_stop (int, optional):Define early stoping patience. Defaults to 25.
        epochs (int, optional): Define number of training epochs. Defaults to 250.
        input_channels (int, optional): Define number of data input channels. Defaults to 10.
        output_channels (int, optional): Define number of model output channels. Defaults to 1.
        input_size (int, optional): Define data input size. Defaults to 128.
        lr (float, optional): Define optimizer learning rate. Defaults to 0.001.
        lr_scheduler (str, optional): Define learning rate scheduler. Options: [None, 'reduce_on_plateau', 'cosine_annealing']. Defaults to None.
        n_shot (int, optional): Define dataset protocol - n samples per region. Defaults to None.
        split_ratio (float, optional): Define dataset protocol - percentage of full dataset. Defaults to 0.1.
        regions (list, optional): Select regions to include in training and test sets. If no regions are defined (None) all avalible regions will be included
                                  Options: [None, 'denmark-1', 'denmark-2', 'east-africa', 'egypt-1', 'eq-guinea', 'europe', 'ghana-1',
                                 'isreal-1', 'isreal-2', 'japan', 'nigeria', 'north-america', 'senegal', 'south-america',
                                 'tanzania-1', 'tanzania-2', 'tanzania-3', 'tanzania-4', 'tanzania-5', 'uganda-1'] Defaults to None.
        vis_val (bool, optional): If set to True data visulisations will be generated at each validation step. Defaults to True.
        warmup (bool, optional): If set to True a linear optimizer warmup phase will occour. Defaults to False.
        warmp_steps (int, optional): Define number of steps for linear warmup phase. Defaults to 5.
        warmup_gamma (int, optional): Define learning rate increase per step in linear warmup phase - new_lr = lr*gamma. Defaults to 10. N.B. initial lr is calulated as follows init_lr = lr/(gamma**warmup_steps)
        pretrained_model_path (str, optional): For pretrained models define the model weights path. Defaults to None.
        freeze_pretrained (bool, optional): If True pretrained encoder weights will be frozen during training. Defaults to None.
        data_path_128_10m (str, optional): Define data path for 128x128 10m resolution dataset. Defaults to None.
        data_path_224_10m (str, optional): Define data path for 224x224 10m resolution dataset. Defaults to None.
        data_path_224_30m (str, optional): Define data path for 224x224 30m resolution dataset. Defaults to None.
        output_path (str, optional): Define folder to save artifacts in. Defaults to None.
        data_parallel (bool, optional): If set to True Model training will be parallized on multiple gpus. Defaults to False.
        device_ids (list, optional): Define GPU IDs to use for parallization. Defaults to None.
    """         
    
    init_lr = lr
    # device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #torch.set_default_device(model_device)
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print('DEVICE', model_device)

    assert not (n_shot == None) or not (split_ratio == None), 'Please define data partition protocol!'
    assert isinstance(n_shot, int) ^ isinstance(split_ratio, float), 'n_shot cannot be used with split_ratio!'
    if (downstream_task == 'lc') or (downstream_task == 'lc_classification'):
        assert (output_channels == 11), 'land cover tasks should have 11 output channels'

    if (downstream_task == 'roads') or (downstream_task == 'building'):
        assert output_channels == 1, 'road and building density estimation tasks should have a single output channel'

    if downstream_task == 'building_classification':
        assert output_channels == 5, 'building classification tasks should have a 5 output channels'

    if downstream_task == 'roads_classification':
        assert output_channels == 2, 'road classification tasks should have a 5 output channels'

    if pretrained_model_path is not None:
        print(model_name)
        assert model_name in (CNN_PRETRAINED_LIST + VIT_CNN_PRETRAINED_LIST), f"Pretrained weights were given but model {model_name} not found in list of pretrained models: {(CNN_PRETRAINED_LIST + VIT_CNN_PRETRAINED_LIST)}"
        assert freeze_pretrained is not None, f"When supplying a pretrained model 'freeze_pretrained' must be either True or False"
        model = get_models_pretrained(model_name, input_channels, output_channels, input_size, path_model_weights=pretrained_model_path, freeze=freeze_pretrained)
        if model_name == 'GeoAware_contrastive_core_nano' or model_name == 'GeoAware_contrastive_core_nano_classifier':
            NAME = model.__class__.__name__ +'_contrastive_frozen' if freeze_pretrained else model.__class__.__name__ +'_contrastive_unfrozen'
        elif model_name == 'GeoAware_mh_pred_core_nano' or model_name == 'GeoAware_mh_pred_core_nano_classifier':
            NAME = model.__class__.__name__ +'_mh_pred_frozen' if freeze_pretrained else model.__class__.__name__ +'_mh_pred_unfrozen'
        else:
            NAME = model.__class__.__name__ + '_frozen' if freeze_pretrained else model.__class__.__name__ + '_unfrozen'

    else:
        if freeze_pretrained:
            print(f"Ignoring freeze_pretrained set to {freeze_pretrained} as no pretrained model was supplied")
        model = get_models(model_name, input_channels, output_channels, input_size)
        NAME = model.__class__.__name__

    OUTPUT_FOLDER = f'{output_path}/{experiment_name}/{downstream_task}/{date.today().strftime("%d%m%Y")}_{NAME}_{downstream_task}'
    if lr_scheduler is not None:
        OUTPUT_FOLDER = f'{output_path}/{experiment_name}/{downstream_task}/{date.today().strftime("%d%m%Y")}_{NAME}_{downstream_task}_{lr_scheduler}'

    if warmup:
        lr = lr / int((warmup_gamma)**(warmp_steps))  # for warmup start

    dataset_folder = data_path_128_10m
    dataset_name = '128_10m'
    if model_name in MODELS_224_r30:
        dataset_folder = data_path_224_30m
        dataset_name = '224_30m'
    if model_name in MODELS_224:
        dataset_folder = data_path_224_10m
        dataset_name = '224_10m'

    if downstream_task == 'pretraining':
        OUTPUT_FOLDER = f'{OUTPUT_FOLDER}'
        x_train, y_train, x_val, y_val = data_protocol.protocol_minifoundation(
            folder='/home/phimultigpu/phileo_NFS/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/',
            y='geo')

        downstream_task = 'geo'

    elif isinstance(n_shot, int):
        OUTPUT_FOLDER = f'{OUTPUT_FOLDER}_{n_shot}'

        x_train, y_train, x_val, y_val = data_protocol.protocol_fewshot_memmapped(
            folder=dataset_folder,
            dst='/home/phimultigpu/phileo_NFS/phileo_data/downstream/downstream_datasets_nshot/',
            n=n_shot,
            regions=regions,
            y=downstream_task,
            data_selection='create',
            name=dataset_name)

    elif isinstance(split_ratio, float): 
        OUTPUT_FOLDER = f'{OUTPUT_FOLDER}_{split_ratio}'
        x_train, y_train, x_val, y_val = data_protocol.protocol_split(
            dataset_folder,
            split_percentage=split_ratio,
            regions=regions,
            y=downstream_task)

    x_test, y_test = data_protocol.get_testset(folder=dataset_folder,
                                               y=downstream_task)

    dl_train, dl_test, dl_val = load_data.load_data(x_train, y_train, x_val, y_val, x_test, y_test,
                                                    with_augmentations=augmentations,
                                                    num_workers=num_workers,
                                                    batch_size=batch_size,
                                                    downstream_task=downstream_task,
                                                    model_name=model_name.split('_')[0],
                                                    device=generator_device
                                                    )


    
    # def weights_init(m, size=0.001):    
    #     """
    #     Initialise the weights of a module.
    #     Does not change the default initialisation method of linear, conv2d, or conv2dtranspose layers.

    #     Parameters
    #     ----------
    #     m : torch.nn.Module
    #         Module to initialise

    #     size : float
    #         Standard deviation of the normal distribution to sample initial values from.
    #         default: 0.001

    #     Returns
    #     -------
    #     None
    #     """

    #     if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
    #         # nn.init.trunc_normal_(m.weight, 1.0, size)

    #         # while torch.any(m.weight == 0.0):
    #         #     nn.init.trunc_normal_(m.weight, 1.0, size)

    #         if m.bias is not None:
    #             nn.init.trunc_normal_(m.bias, 0.0, size)

    #             while torch.any(m.bias == 0.0):
    #                 nn.init.trunc_normal_(m.bias, 0.0, size)

    #     if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)) and m.bias is not None:
    #         nn.init.trunc_normal_(m.bias, 0.0, size)

    #         while torch.any(m.bias == 0.0):
    #             nn.init.trunc_normal_(m.bias, 0.0, size)


    # class GaussianDropout2d(nn.Module):
    #     """
    #     Drop out channels of a 2D input with Gaussian noise.

    #     Parameters
    #     ----------
    #     p : float
    #         Probability of dropping a channel
    #         default: 0.5

    #     signal_to_noise : tuple
    #         Range of signal to noise ratios to use for the dropped channels. 0.0 is pure noise, 1.0 is pure signal.
    #         The amount of signal is randomly sampled from this range for each channel.
    #         If None, no signal is added to the dropped channels.
    #         default: (0.1, 0.9)
    #     """

    #     def __init__(self, p=0.5, signal_to_noise=(0.1, 0.9)):
    #         super(GaussianDropout2d, self).__init__()
    #         self.p = p
    #         self.signal_to_noise = signal_to_noise

    #     def forward(self, x):
    #         if self.training:
    #             batch_size, num_channels, height, width = x.size()

    #             # Create a mask of channels to drop
    #             mask = torch.rand(batch_size, num_channels, 1, 1, device=x.device) > self.p

    #             # If all channels are dropped, redraw the mask
    #             while torch.all(mask):
    #                 mask = torch.rand(batch_size, num_channels, 1, 1, device=x.device) > self.p

    #             mean = x.mean([2, 3], keepdim=True).repeat(1, 1, height, width)
    #             std = x.std([2, 3], keepdim=True).repeat(1, 1, height, width)

    #             # Create the noise (Same mean and std as the input)
    #             noise = torch.normal(mean, torch.clamp(std, min=1e-6))

    #             if self.signal_to_noise is not None:
    #                 signal_level = torch.rand(batch_size, num_channels, 1, 1, device=x.device) * (
    #                         self.signal_to_noise[1] - self.signal_to_noise[0]) + self.signal_to_noise[0]
    #                 adjusted_noise = noise * (1 - signal_level)
    #                 adjusted_signal = x * signal_level

    #             # Combine the adjusted noise and signal
    #             return (adjusted_signal * mask) + (adjusted_noise * (~mask))

    #         return x

    # class GaussianDropout1d(nn.Module):   
    #     def __init__(self, p=0.5): 
    #         super(GaussianDropout1d, self).__init__()
    #         self.p = p

    #     def forward(self, x):
    #         if self.training:
    #             batch_size, size = x.size()

    #             # Create a mask of channels to drop
    #             mask = torch.rand(batch_size, size, device=x.device) > self.p

    #             # If all channels are dropped, redraw the mask
    #             while torch.all(mask):
    #                 mask = torch.rand(batch_size, size, device=x.device) > self.p

    #             mean = x.mean([1], keepdim=True).repeat(1, size)
    #             std = x.std([1], keepdim=True).repeat(1, size)

    #             # Create the noise (Same mean and std as the input)
    #             noise = torch.normal(mean, torch.clamp(std, min=1e-6))

    #             # Combine the adjusted noise and signal
    #             return (x * mask) + (noise * (~mask))

    #         return x

    # class RandomMask2D(nn.Module):   
    #     """
    #     Randomly masks pixels of an image with zeros across all channels

    #     Parameters
    #     ----------
    #     p : float
    #         Probability of masking a pixel
    #         default: 0.5
    #     """

    #     def __init__(self, p=0.5):
    #         super(RandomMask2D, self).__init__()
    #         self.p = p

    #     def forward(self, x):
    #         if self.training:
    #             mask = torch.rand(x.size(0), 1, x.size(2), x.size(3), device=x.device) > self.p

    #             return x * mask

    #         return x

    # class ScaleSkip2D(nn.Module):  
    #     """
    #     Learnable channel-wise scale and bias for skip connections.

    #     Parameters
    #     ----------
    #     channels : int
    #         Number of channels in the input

    #     drop_y : float
    #         Probability of dropping a channel in the skip connection.
    #         Drops are replaces with Gaussian noise.

    #     signal_to_noise : tuple or None
    #         Range of signal to noise ratios to use for the dropped channels. 0.0 is pure noise, 1.0 is pure signal.
    #         The amount of signal is randomly sampled from this range for each channel.
    #         If None, no signal is added to the dropped channels.
    #         default: (0.1, 0.9)

    #     size : float
    #         Standard deviation of the normal distribution to sample initial values from.
    #         default: 0.01
    #     """

    #     def __init__(self, channels, drop_y=None, signal_to_noise=(0.1, 0.9), size=0.01):
    #         super(ScaleSkip2D, self).__init__()
    #         self.channels = channels
    #         self.drop_y = drop_y
    #         self.size = size

    #         # Learnable scale and bias
    #         self.x_skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1))
    #         self.y_skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1))
    #         self.y_skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
    #         self.x_skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))

    #         if self.drop_y is not None and self.drop_y > 0.0:
    #             self.drop_y = GaussianDropout2d(self.drop_y, signal_to_noise=signal_to_noise)
    #         else:
    #             self.drop_y = None

    #         self.set_weights()
    #         while torch.any(self.x_skipscale == 0) or torch.any(self.y_skipscale == 0) or torch.any(
    #                 self.y_skipbias == 0
    #         ) or torch.any(self.x_skipbias == 0):
    #             self.set_weights()

    #     def set_weights(self):
    #         nn.init.trunc_normal_(self.x_skipscale, 1.0, self.size)
    #         nn.init.trunc_normal_(self.y_skipscale, 1.0, self.size)
    #         nn.init.trunc_normal_(self.y_skipbias, 0.0, self.size)
    #         nn.init.trunc_normal_(self.x_skipbias, 0.0, self.size)

    #     def forward(self, x, y):
    #         x = (x * self.x_skipscale) + self.x_skipbias
    #         y = (y * self.y_skipscale) + self.y_skipbias

    #         if self.drop_y is not None:
    #             y = self.drop_y(y)

    #         return x + y

    # class ScaleSkip1D(nn.Module):  
    #     """ Learnable weight and bias for 1D skip connections. """

    #     def __init__(self, drop_y=None, size=0.01):
    #         super(ScaleSkip1D, self).__init__()

    #         self.size = size
    #         self.drop_y = drop_y

    #         # Learnable scale and bias
    #         self.x_skipscale = nn.Parameter(torch.ones(1, 1))
    #         self.y_skipscale = nn.Parameter(torch.ones(1, 1))
    #         self.y_skipbias = nn.Parameter(torch.zeros(1, 1))
    #         self.x_skipbias = nn.Parameter(torch.zeros(1, 1))

    #         self.set_weights()
    #         while torch.any(self.x_skipscale == 0) or torch.any(self.y_skipscale == 0) or torch.any(
    #                 self.y_skipbias == 0
    #         ) or torch.any(self.x_skipbias == 0):
    #             self.set_weights()

    #         if self.drop_y is not None and self.drop_y > 0.0:
    #             self.drop_y = GaussianDropout1d(self.drop_y)
    #         else:
    #             self.drop_y = None

    #     def set_weights(self):
    #         nn.init.trunc_normal_(self.x_skipscale, 1.0, self.size)
    #         nn.init.trunc_normal_(self.y_skipscale, 1.0, self.size)
    #         nn.init.trunc_normal_(self.y_skipbias, 0.0, self.size)
    #         nn.init.trunc_normal_(self.x_skipbias, 0.0, self.size)

    #     def forward(self, x, y):
    #         x = (x * self.x_skipscale) + self.x_skipbias
    #         y = (y * self.y_skipscale) + self.y_skipbias

    #         if self.drop_y is not None:
    #             y = self.drop_y(y)

    #         return x + y

    # class SE_Block(nn.Module):   
    #     """ credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4 """

    #     def __init__(self, channels, reduction=16):
    #         super().__init__()
    #         self.reduction = reduction
    #         self.squeeze = nn.AdaptiveAvgPool2d(1)
    #         self.excitation = nn.Sequential(
    #             nn.Linear(channels, max(1, channels // self.reduction), bias=False),
    #             nn.GELU(),
    #             nn.Linear(max(1, channels // self.reduction), channels, bias=False),
    #             nn.Sigmoid()
    #         )

    #     def forward(self, x):
    #         bs, c, _, _ = x.shape
    #         y = self.squeeze(x).view(bs, c)
    #         y = self.excitation(y).view(bs, c, 1, 1)

    #         return x * y.expand_as(x)

    # class CNNBlock(nn.Module):  
    #     """
    #     This is a standard CNN block with a 1x1 convolutional matcher for the skip connection.
    #     It adds a learnable scale and bias to the skip connection.

    #     Parameters
    #     ----------
    #     channels_in : int
    #         Number of channels in the input

    #     channels_out : int or None
    #         Number of channels in the output. If None, the number of channels is unchanged.
    #         default: None

    #     group_size : int
    #         Number of groups for the 3x3 convolution.
    #         default: 1

    #     activation : torch.nn.Module
    #         Activation function to use after the first convolution.
    #         default: torch.nn.GELU()

    #     activation_out : torch.nn.Module or None
    #         Activation function to use after the last convolution.
    #         If None, the same activation as the first convolution is used.
    #         default: None

    #     chw : tuple or None
    #         Height and width of the input. If None, batch norm is used instead of layer norm.
    #         default: None
    #     """

    #     def __init__(
    #             self,
    #             channels_in,
    #             channels_out=None,
    #             chw=None,
    #             group_size=1,
    #             activation=nn.GELU(),
    #             activation_out=None,
    #             residual=True,
    #             reduction=1,
    #     ):
    #         super().__init__()

    #         assert chw is not None, "chw must be specified"

    #         self.channels_in = channels_in
    #         self.channels_out = channels_in if channels_out is None else channels_out
    #         self.channels_internal = self.channels_out // reduction
    #         self.chw = chw
    #         self.group_size = group_size
    #         self.activation = activation
    #         self.activation_out = activation if activation_out is None else activation_out
    #         self.residual = residual
    #         self.reduction = reduction
    #         self.squeeze = SE_Block(self.channels_out, 16)

    #         self.matcher = nn.Conv2d(
    #             self.channels_in, self.channels_out, 1, padding=0,
    #             bias=False
    #             ) if self.channels_in != self.channels_out else None

    #         self.norm1 = nn.LayerNorm([self.channels_internal, self.chw[1], self.chw[2]])
    #         self.norm2 = nn.LayerNorm([self.channels_internal, self.chw[1], self.chw[2]])

    #         self.conv1 = nn.Conv2d(self.channels_in, self.channels_internal, 1, padding=0, bias=False)
    #         self.conv2 = nn.Conv2d(
    #             self.channels_internal, self.channels_internal, 3, padding=1, groups=self.group_size,
    #             bias=False, padding_mode="replicate"
    #             )
    #         self.conv3 = nn.Conv2d(self.channels_internal, self.channels_out, 1, padding=0, bias=True)

    #         self.scaler = ScaleSkip2D(self.channels_out) if self.residual else None

    #     def forward(self, x):
    #         identity = x if self.matcher is None else self.matcher(x)

    #         x = self.conv1(x)
    #         x = self.norm1(x)
    #         x = self.activation(x)

    #         x = self.conv2(x)
    #         x = self.norm2(x)
    #         x = self.activation(x)

    #         x = self.conv3(x)
    #         x = self.squeeze(x)

    #         if self.residual:
    #             x = self.scaler(x, identity)

    #         x = self.activation_out(x)
    #         return x

    # class GlobalBlock(nn.Module):  
    #     """
    #     Global Block for the paper `'Global Context Dynamic-CNNs (Fibaek et al., 2024)'`

    #     Parameters
    #     ----------
    #     in_channels : int
    #         Number of input channels

    #     out_channels : int or None
    #         Number of output channels. If None, the number of channels is unchanged.
    #         default: None

    #     kernel_size : int
    #         Size of the second convolutional kernel.
    #         default: 3

    #     patch_size : int
    #         Size of the patches to split the image into.
    #         default: 16

    #     chw : tuple
    #         Height and width of the input. Must be divisible by patch_size.
    #         default: None

    #     activation : torch.nn.Module
    #         Activation function to use after the first convolution.
    #         default: torch.nn.GELU()

    #     activation_out : torch.nn.Module or None
    #         Activation function to use after the last convolution.
    #         If None, the same activation as the first convolution is used.
    #         default: None

    #     reduction : int
    #         Reduction factor for the internal channels.
    #         default: 1 (no reduction)

    #     residual : bool
    #         Whether to use a residual connection.
    #         default: True

    #     patch_dim : int
    #         Dimension of the patch embeddings.
    #         default: 512

    #     projector : torch.nn.Module or None
    #         Projector to use for the patch embeddings. If None, a new projector is created.
    #         ```python
    #         self.projector = nn.Sequential(
    #             nn.Linear(self.in_channels * (self.patch_size ** 2), self.patch_dim),
    #             nn.LayerNorm(self.patch_dim),
    #             nn.GELU(),
    #         )
    #         ```
    #         default: None
    #     """

    #     def __init__(
    #             self,
    #             in_channels,
    #             out_channels=None,
    #             kernel_size=3,
    #             patch_size=16,
    #             chw=None,
    #             activation=nn.GELU(),
    #             activation_out=None,
    #             reduction=1,
    #             residual=True,
    #             patch_dim=512,
    #             shared_context=32,
    #             num_heads=8,
    #             projector=None,
    #     ):
    #         super(GlobalBlock, self).__init__()
    #         self.in_channels = in_channels
    #         self.out_channels = out_channels if out_channels is not None else in_channels
    #         self.kernel_size = kernel_size
    #         self.patch_size = patch_size
    #         self.chw = chw
    #         self.activation = activation
    #         self.activation_out = activation_out if activation_out is not None else activation
    #         self.reduction = reduction
    #         self.residual = residual
    #         self.patch_dim = patch_dim
    #         self.shared_context = shared_context
    #         self.num_heads = num_heads
    #         self.projector = projector

    #         assert chw is not None, "chw must be specified"
    #         assert chw[1] == chw[2], "chw must be square"
    #         assert chw[1] % patch_size == 0, "chw must be divisible by patch_size"
    #         assert chw[1] >= patch_size, "patch_size must be greater than or equal to chw"

    #         self.num_patches_height = self.chw[1] // self.patch_size
    #         self.num_patches_width = self.chw[2] // self.patch_size
    #         self.num_patches = self.num_patches_height * self.num_patches_width

    #         self.internal_channels = self.out_channels // self.reduction

    #         self.latent_1x1 = self.out_channels * self.internal_channels + self.internal_channels
    #         self.latent_3x3 = \
    #             self.internal_channels * self.internal_channels * kernel_size * kernel_size + self.internal_channels
    #         self.latent_1x1_out = self.internal_channels * self.out_channels + self.out_channels
    #         self.context_size = (self.patch_dim * self.num_patches) + (self.shared_context * 3)

    #         self.projector = nn.Linear(
    #             self.out_channels * (self.patch_size ** 2),
    #             self.patch_dim
    #             ) if projector is None else projector

    #         self.conv_1x1 = nn.Linear(self.context_size, self.latent_1x1 + self.shared_context)
    #         self.conv_3x3 = nn.Linear(self.context_size, self.latent_3x3 + self.shared_context)
    #         self.conv_1x1_out = nn.Linear(self.context_size, self.latent_1x1_out + self.shared_context)

    #         self.multihead_attn = nn.MultiheadAttention(
    #             embed_dim=self.patch_dim, num_heads=self.num_heads,
    #             batch_first=True, add_zero_attn=True, add_bias_kv=True
    #             )

    #         self.scaler = ScaleSkip2D(self.out_channels)
    #         self.scaler_ctx = ScaleSkip1D()

    #         self.pos_embed = self.posemb_sincos_2d(self.num_patches_height, self.num_patches_width, self.patch_dim)

    #         # # So many normalisations - so little time
    #         self.norm_input = nn.LayerNorm([self.out_channels, self.chw[1], self.chw[2]])
    #         self.norm_patches1 = nn.LayerNorm([self.num_patches, self.patch_dim])
    #         self.norm_patches2 = nn.LayerNorm([self.num_patches, self.patch_dim])
    #         self.context_norm_1x1 = nn.LayerNorm(self.context_size)
    #         self.context_norm_3x3 = nn.LayerNorm(self.context_size)
    #         self.context_norm_1x1_out = nn.LayerNorm(self.context_size)
    #         self.context_shared_norm = nn.LayerNorm((self.shared_context * 3))
    #         self.cnn_norm_1x1 = nn.LayerNorm([self.internal_channels, self.chw[1], self.chw[2]])
    #         self.cnn_norm_3x3 = nn.LayerNorm([self.internal_channels, self.chw[1], self.chw[2]])

    #         if self.in_channels == self.out_channels: 
    #             self.matcher = nn.Identity()
    #         else:
    #             self.matcher = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)

    #         self.apply(weights_init)

    #     def patchify_batch(self, tensor): 
    #         """
    #         Split a batch of images into patches. 

    #         Parameters
    #         ----------
    #         tensor : torch.Tensor
    #             Batch of images to split into patches
    #             Shape: `(B, C, H, W)`

    #         Returns
    #         -------
    #         torch.Tensor
    #             Batch of patches
    #             Shape: `(B, num_patches, C * (patch_size ** 2))`
    #         """
    #         B, C, _H, _W = tensor.shape

    #         reshaped = tensor.reshape(
    #             B, C, self.num_patches_height, self.patch_size, self.num_patches_width,
    #             self.patch_size
    #             )
    #         transposed = reshaped.permute(0, 2, 4, 1, 3, 5)
    #         patches = transposed.reshape(B, self.num_patches, C * self.patch_size * self.patch_size)
    #         return patches

    #     def convolve(self, x, context, in_channels, out_channels, size=3):
    #         """
    #         Perform a convolution with a learned context (kernel) vector.

    #         Parameters
    #         ----------
    #         x : torch.Tensor
    #             Input tensor
    #             Shape: `(B, C, H, W)`

    #         context : torch.Tensor
    #             Context vector, reshaped into a kernel
    #             Shape: `(B, out_channels, in_channels, size, size)`

    #         in_channels : int
    #             Number of input channels

    #         out_channels : int
    #             Number of output channels

    #         size : int
    #             Size of the kernel
    #             default: 3

    #         Returns
    #         -------
    #         torch.Tensor
    #             Convolved output tensor
    #             Shape: `(B, out_channels, H, W)`
    #         """

    #         batch_size = x.size(0)

    #         _kernel, _bias = torch.split(context, [context.size(1) - out_channels, out_channels], dim=1)
    #         kernel = _kernel.reshape(batch_size * out_channels, in_channels, size, size)
    #         bias = _bias.reshape(batch_size * out_channels)

    #         x = x.reshape(1, batch_size * in_channels, x.shape[2], x.shape[3])
    #         if size != 1:
    #             x = F.pad(x, (self.kernel_size // 2,) * 4)

    #         x = F.conv2d(x, kernel, groups=batch_size, bias=bias)
    #         x = x.reshape(batch_size, out_channels, x.shape[2], x.shape[3])
    #         return x

    #     @staticmethod
    #     def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    #         assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    #         y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    #         omega = torch.arange(dim // 4) / (dim // 4 - 1)
    #         omega = 1.0 / (temperature ** omega)

    #         y = y.flatten()[:, None] * omega[None, :]
    #         x = x.flatten()[:, None] * omega[None, :]

    #         pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    #         return pe.type(dtype)

    #     def forward_stem(self, x):
    #         x = self.patchify_batch(x)  # shape: (B, num_patches, C * (patch_size ** 2))
    #         x = self.projector(x)  # shape: (B, num_patches, patch_dim)
    #         x = self.norm_patches1(x)
    #         attn, _ = self.multihead_attn(x, x, x)  # self-attention
    #         x = x + attn
    #         x = self.norm_patches2(x)
    #         x = x + self.pos_embed.to(x.device)
    #         x = x.reshape(x.shape[0], -1)  # shape: (B, num_patches * patch_dim)
    #         return x

    #     def forward(self, x, ctx_p=None):
    #         identity = x if self.in_channels == self.out_channels else self.matcher(x)  # shape: (B, C, H, W)
    #         x = self.norm_input(identity)

    #         ctx_p = torch.zeros(
    #             (x.shape[0], self.shared_context * 3),
    #             device=x.device
    #             ) if ctx_p is None else ctx_p  # shape: (B, shared_context)
    #         prev_1x1, prev_3x3, prev_1x1_out = ctx_p.split(
    #             [self.shared_context, self.shared_context, self.shared_context],
    #             dim=1
    #             )

    #         embedded_patches = self.forward_stem(x)  # shape: (B, num_patches * patch_dim)

    #         # 1x1 Convolution with global context
    #         inputs = self.context_norm_1x1(torch.cat([embedded_patches, prev_1x1, prev_3x3, prev_1x1_out], dim=1))
    #         combined_context = self.conv_1x1(inputs)
    #         ctx, ctx_1x1 = combined_context.split([self.latent_1x1, self.shared_context], dim=1)
    #         x = self.convolve(x, ctx, self.out_channels, self.internal_channels, size=1)
    #         x = self.cnn_norm_1x1(x)

    #         # 3x3 Convolution with global context
    #         inputs = self.context_norm_3x3(torch.cat([embedded_patches, ctx_1x1, prev_3x3, prev_1x1_out], dim=1))
    #         combined_context = self.conv_3x3(inputs)
    #         ctx, ctx_3x3 = combined_context.split([self.latent_3x3, self.shared_context], dim=1)
    #         x = self.convolve(x, ctx, self.internal_channels, self.internal_channels, size=self.kernel_size)
    #         x = self.cnn_norm_3x3(x)

    #         # 1x1 Convolution with global context
    #         inputs = self.context_norm_1x1_out(torch.cat([embedded_patches, ctx_1x1, ctx_3x3, prev_1x1_out], dim=1))
    #         combined_context = self.conv_1x1_out(inputs)
    #         ctx, ctx_1x1_out = combined_context.split([self.latent_1x1_out, self.shared_context], dim=1)
    #         x = self.convolve(x, ctx, self.internal_channels, self.out_channels, size=1)

    #         # Merge contexts
    #         ctx_o = self.context_shared_norm(torch.cat([ctx_1x1, ctx_3x3, ctx_1x1_out], dim=1))

    #         # Learned skip-connection
    #         x = self.scaler(x, identity) if self.residual else x
    #         ctx_o = self.scaler_ctx(ctx_o, ctx_p) if self.residual else ctx_o

    #         # Activation
    #         x = self.activation_out(x)
    #         ctx_o = self.activation_out(ctx_o)

    #         return x, ctx_o

    # class FoundationEncoder(nn.Module):
    #     def __init__(
    #             self,
    #             *,
    #             input_dim=3,
    #             depths=None,
    #             dims=None,
    #             img_size=64,
    #             latent_dim=512,
    #             activation=nn.LeakyReLU(),
    #     ):
    #         super().__init__()

    #         self.depths = depths  
    #         self.dims = dims
    #         self.input_dim = input_dim
    #         self.img_size = img_size
    #         self.latent_dim = latent_dim
    #         self.steps = 1
    #         self.sizes = [img_size]
    #         self.activation = activation

    #         for i in range(len(self.depths) - 1):
    #             half = self.sizes[-1] // 2
    #             self.sizes.append(half)
    #             self.steps += 1

    #         self.linear_dim = int(((img_size // (2 ** (self.steps - 1))) ** 2) * self.dims[-1])

    #         assert len(self.depths) == self.steps, "Invalid depths"
    #         assert len(self.dims) == self.steps, "Invalid dims"
    #         assert self.depths is not None, "Invalid depths"
    #         assert self.dims is not None, "Invalid dims"
    #         assert self.steps == len(self.dims), "Invalid dims"

    #         self.downsample = nn.ModuleList()
    #         for i in range(self.steps - 1):
    #             self.downsample.append(
    #                 nn.Sequential(
    #                     nn.Conv2d(self.dims[i], self.dims[i + 1], 1, padding=0),
    #                     nn.MaxPool2d(2, stride=2),
    #                 )
    #             )

    #         self.block_scalers = nn.ModuleList()
    #         for i in range(self.steps):
    #             self.block_scalers.append(ScaleSkip2D(self.dims[i]))

    #         self.blocks_down = nn.ModuleList()
    #         for i in range(self.steps):
    #             self.blocks_down.append(nn.ModuleList())
    #             for _ in range(self.depths[i]):
    #                 self.blocks_down[i].append(
    #                     CNNBlock(self.dims[i], chw=[self.dims[i], self.sizes[i], self.sizes[i]], activation=self.activation)
    #                 )

    #         self.prelinear_norm = nn.LayerNorm([self.dims[-1], self.sizes[-1], self.sizes[-1]])
    #         self.linear_encode = nn.Sequential(
    #             self.activation,
    #             nn.Linear(self.linear_dim, self.latent_dim),
    #             nn.LayerNorm(self.latent_dim),
    #         )

    #         self.head_clouds = nn.Sequential(
    #             nn.Linear(self.latent_dim, 4),
    #         )

    #         self.head_landcover = nn.Sequential(
    #             nn.Linear(self.latent_dim, 11),
    #         )

    #         self.head_buildings = nn.Sequential(
    #             nn.Linear(self.latent_dim, 1),
    #             nn.Sigmoid(),
    #         )

    #         self.head_coords = nn.Sequential(
    #             nn.Linear(self.latent_dim, 4),
    #             nn.Sigmoid(),
    #         )

    #     def forward(self, x):
    #         skips = []

    #         for i in range(self.steps):
    #             pre_block = x
    #             for j in range(self.depths[i]):
    #                 block = self.blocks_down[i][j]
    #                 x = block(x)

    #             if len(self.blocks_down[i]) > 1:
    #                 x = self.block_scalers[i](x, pre_block)

    #             skips.append(x)

    #             if i < self.steps - 1:
    #                 x = self.downsample[i](x)

    #         embeddings_cnn = self.prelinear_norm(x)
    #         flat = embeddings_cnn.reshape(-1, self.linear_dim)
    #         embeddings = self.linear_encode(flat)
    #         out_coords = self.head_coords(embeddings)  # 4
    #         out_clouds = self.head_clouds(embeddings)  # 4
    #         out_buildings = self.head_buildings(embeddings)
    #         out_landcover = self.head_landcover(embeddings)

    #         return (
    #             embeddings,
    #             embeddings_cnn,
    #             skips,
    #             (
    #                 out_coords,
    #                 out_clouds,
    #                 out_buildings,
    #                 out_landcover,
    #             )
    #         )

    # class FoundationDecoder(nn.Module):  
    #     def __init__(
    #             self,
    #             *,
    #             depths=None,
    #             dims=None,
    #             img_size=64,
    #             latent_dim=512,
    #             dropout=None,
    #             activation=nn.LeakyReLU(),
    #     ):
    #         super().__init__()
    #         self.depths = depths
    #         self.dims = dims
    #         self.img_size = img_size
    #         self.latent_dim = latent_dim
    #         self.steps = 1
    #         self.sizes = [img_size]
    #         self.dropout = dropout
    #         self.activation = activation

    #         for i in range(len(self.depths) - 1):
    #             half = self.sizes[-1] // 2
    #             self.sizes.append(half)
    #             self.steps += 1

    #         self.sizes = self.sizes[::-1]
    #         self.linear_dim = int(((img_size // (2 ** (self.steps - 1))) ** 2) * self.dims[0])

    #         if self.dropout is None:
    #             self.dropout = [0.0] * self.steps
    #         elif isinstance(self.dropout, (int, float)):
    #             self.dropout = [self.dropout] * self.steps

    #         assert len(self.depths) == self.steps, "Invalid depths"
    #         assert len(self.dims) == self.steps, "Invalid dims"
    #         assert len(self.dropout) == self.steps, "Invalid dropout"
    #         assert self.depths is not None, "Invalid depths"
    #         assert self.dims is not None, "Invalid dims"
    #         assert self.dropout is not None, "Invalid dropout"

    #         self.linear_decode = nn.Linear(self.latent_dim, self.linear_dim)

    #         self.latent_norm = nn.LayerNorm(
    #             [self.dims[0], self.img_size // (2 ** (self.steps - 1)), self.img_size // (2 ** (self.steps - 1))]
    #         )
    #         self.prehead_norm = nn.LayerNorm([self.dims[-1], self.sizes[-1], self.sizes[-1]])

    #         self.skip_scalers = nn.ModuleList()
    #         self.block_scalers = nn.ModuleList()
    #         for i in range(self.steps):
    #             self.skip_scalers.append(ScaleSkip2D(self.dims[i], drop_y=self.dropout[i], signal_to_noise=(0.1, 0.9)))
    #             self.block_scalers.append(ScaleSkip2D(self.dims[i]))

    #         self.blocks_up = nn.ModuleList()
    #         for i in range(self.steps):
    #             self.blocks_up.append(nn.ModuleList())
    #             for _ in range(self.depths[i]):
    #                 self.blocks_up[i].append(
    #                     CNNBlock(self.dims[i], chw=[self.dims[i], self.sizes[i], self.sizes[i]], activation=self.activation)
    #                 )

    #         self.upsamplers = nn.ModuleList()
    #         for i in range(self.steps - 1):
    #             self.upsamplers.append(
    #                 nn.Sequential(
    #                     nn.UpsamplingBilinear2d(scale_factor=2),
    #                     nn.Conv2d(self.dims[i], self.dims[i + 1], 3, padding=1, bias=False, padding_mode='replicate'),
    #                     nn.LayerNorm([self.dims[i + 1], self.sizes[i + 1], self.sizes[i + 1]]),
    #                     self.activation,
    #                 )
    #             )

    #     def forward(self, x, skips):
    #         x = self.linear_decode(x)
    #         x = x.reshape(
    #             -1,
    #             self.dims[0],
    #             self.img_size // (2 ** (self.steps - 1)),
    #             self.img_size // (2 ** (self.steps - 1))
    #         )
    #         x = self.latent_norm(x)

    #         for i in range(self.steps):
    #             skip_x = skips[-(i + 1)]
    #             x = self.skip_scalers[i](x, skip_x)

    #             pre_block = x
    #             for block in self.blocks_up[i]:
    #                 x = block(x)

    #             if len(self.blocks_up[i]) > 1:
    #                 x = self.block_scalers[i](x, pre_block)

    #             if i < self.steps - 1:
    #                 x = self.upsamplers[i](x)

    #         x = self.prehead_norm(x)
    #         return x

    # class Foundation(nn.Module):
    #     def __init__(
    #             self,
    #             *,
    #             input_dim=3,
    #             output_dim=None,
    #             depths=None,
    #             dims=None,
    #             img_size=64,
    #             latent_dim=512,
    #             dropout=None,
    #             activation=nn.LeakyReLU(),
    #     ):
    #         super().__init__() 

    #         self.input_dim = input_dim
    #         self.output_dim = input_dim if output_dim is None else output_dim
    #         self.depths = depths
    #         self.dims = dims
    #         self.img_size = img_size
    #         self.latent_dim = latent_dim
    #         self.dropout = dropout
    #         self.activation = activation

    #         self.stem = CNNBlock(
    #             input_dim,
    #             dims[0],
    #             chw=[input_dim, img_size, img_size],
    #             activation=self.activation,
    #         )

    #         self.encoder = FoundationEncoder(
    #             input_dim=dims[0],
    #             depths=depths,
    #             dims=dims,
    #             img_size=img_size,
    #             latent_dim=latent_dim,
    #             activation=self.activation,
    #         )

    #         self.decoder = FoundationDecoder(
    #             depths=depths[::-1],
    #             dims=dims[::-1],
    #             img_size=img_size,
    #             latent_dim=latent_dim,
    #             dropout=dropout,
    #             activation=self.activation,
    #         )

    #         # self.head = CNNBlock(
    #         #     self.dims[0],
    #         #     self.output_dim,
    #         #     chw=[self.output_dim, self.img_size, self.img_size],
    #         #     activation=self.activation,
    #         #     #activation_out=nn.Sigmoid(),
    #         #     activation_out=nn.Softmax(),
    #         # )       
            
    #         # self.head = CNNBlock(
    #         #     self.dims[0],
    #         #     11,
    #         #     chw=[self.output_dim, self.img_size, self.img_size],
    #         #     activation=self.activation,
    #         #     #activation_out=nn.Sigmoid(),
    #         #     activation_out=nn.Softmax(),
    #         # )   
            
    #         self.head = nn.Sequential(CNNBlock(
    #             self.dims[0],
    #             self.output_dim,
    #             chw=[self.output_dim, self.img_size, self.img_size],
    #             activation=self.activation,
    #             activation_out=nn.Sigmoid(),
    #             #activation_out=nn.Softmax(dim=11),
    #         ), nn.Conv2d(self.output_dim, 11, kernel_size=1, padding=0)) # # better or worse than 'acc': 0.6996220753605205?       
            
    #         # self.head = nn.Sequential(CNNBlock(
    #         #     self.dims[0],
    #         #     self.dims[0],
    #         #     chw=[self.output_dim, self.img_size, self.img_size],
    #         #     activation=self.activation,
    #         #     #activation_out=nn.Sigmoid(),
    #         #     #activation_out=nn.Softmax(dim=11),
    #         # ), nn.Conv2d(self.dims[0], 11, kernel_size=1, padding=0))  
            
    #         # self.head = nn.Sequential(
    #         #     CNNBlock(self.dims[0], self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding),
    #         #     nn.Conv2d(self.dims[0], self.output_dim, kernel_size=1, padding=0),
    #         # )           

    #     def forward(self, x):
    #         x = self.stem(x)
    #         embeddings, embeddings_cnn, skips, predictions = self.encoder(x)
    #         decoded = self.decoder(embeddings, skips)
    #         reconstruction = self.head(decoded)

    #         #return reconstruction, embeddings, embeddings_cnn, decoded, predictions        
    #         return reconstruction
    # model = Foundation(
    #         input_dim=10,  # B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
    #         # input_dim=13, # B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
    #         #depths=[3, 3, 4, 4, 5],  # 128, 64, 32, 16, 8
    #         #dims=[32, 32, 64, 64, 128],
    #         depths=[3, 3, 12, 12, 5],  
    #         dims=[86, 86, 256, 256, 512],
    #         img_size=128,
    #         latent_dim=1024,
    #         # dropout=[0.85, 0.90, 0.90, 0.95, 0.95], 
    #         dropout=None,
    #         activation=nn.GELU(),
    #     )   


    
    # from model.RS3Mamba import RS3Mamba, load_pretrained_ckpt                            

    # #model = RS3Mamba(num_classes=11).cuda()    
    # model = RS3Mamba(num_classes=1).cuda()    
    
    # #model.eval()  
    # model.train()



    # from Mamba2D.models.mamba2d import Mamba2DBackbone   
    # from Mamba2D.models.utils import MlpHead  
    
    # model = Mamba2DBackbone() 
    
    # # model.train()   



    # # # Create patches and apply flatten operation                 
    # # # With the batch support the operation becomes this in Torch 
    # # # Credit: # # https://discuss.pytorch.org/t/how-to-extract-patches-from-an-image/79923/4
    # #unfolded_batch = image_batch.unfold(1, C, C).unfold(2, P, P).unfold(3, P, P) 
    # C = 10   
    # P = 16 
    # #P = 4 
    # B = 100
    # desired_image_size = 128
    # patch_dim = int(P * P * C)
    # number_of_patches = int((desired_image_size * desired_image_size)/ (P * P))
    # #unfolded_batch = images.unfold(1, C, C).unfold(2, P, P).unfold(3, P, P) 
    # #flattened_patches = unfolded_batch.contiguous().view(B, -1, C * P * P).float()

    # # # Linear Projection process    
    # #patch_weights = nn.Parameter(torch.empty(P * P * C, D).normal_(std=0.02)) 
    # D = 768   
    # patch_weights = nn.Parameter(torch.empty(P * P * C, D).normal_(std=0.02)).cuda()

    # #patch_embeddings = torch.matmul(flattened_patches, patch_weights)
    
    # # # Initialisation of Class Token      
    # #class_token = nn.Parameter(torch.empty(1, 1, D).normal_(std=0.02)) 
    # class_token = nn.Parameter(torch.empty(1, 1, D).normal_(std=0.02)).cuda() 

    # batch_class_token = class_token.expand(B, -1, -1)
    # #patch_embeddings_with_class_token = torch.cat([batch_class_token, patch_embeddings], dim=1)

    # # # Addition of the Positional Embeddings to Patch Embeddings with Class Tokens   
    # #positional_embedding = nn.Parameter(torch.empty(B, number_of_patches + 1, D).normal_(std=0.02))
    # positional_embedding = nn.Parameter(torch.empty(B, number_of_patches + 1, D).normal_(std=0.02)).cuda() 

    # #embeddings = patch_embeddings_with_class_token + positional_embedding

    # # # Self Attention Module      
    # class SelfAttention(nn.Module):
    #     def __init__(self, embedding_dim, qkv_dim):
    #         super(SelfAttention, self).__init__()

    #         self.embedding_dim = embedding_dim   # embedding dimension 
    #         self.qkv_dim = qkv_dim               # Dimension of key, query, value

    #         #self.W = nn.Parameter(torch.empty(1, embedding_dim, int(3 * qkv_dim)).normal_(std=0.02))  
    #         self.W = nn.Parameter(torch.empty(1, embedding_dim, int(3 * qkv_dim)).normal_(std=0.02)).cuda() 

    #     def forward(self, embeddings):
    #         # calculate query, key and value projection
    #         qkv = torch.matmul(embeddings, self.W)
    #         q = qkv[:, :, :self.qkv_dim]
    #         k = qkv[:, :, self.qkv_dim:self.qkv_dim*2 ]
    #         v = qkv[:, :, self.qkv_dim*2:]
            
    #         # Calculate attention weights by applying a softmax to the dot product of all queries with all keys
    #         attention_weights = F.softmax(torch.matmul(q, torch.transpose(k, -2, -1) ) / math.sqrt(self.qkv_dim), dim=1)
            
    #         # calculate attention values and return
    #         return torch.matmul(attention_weights, v)

    # # # initialise self-attention object    
    # #self_attention = SelfAttention(embedding_dim=D, qkv_dim=int(3 * qkv_dim))  
    # num_heads = 12  
    # qkv_dim = int(D / num_heads)
    # self_attention = SelfAttention(embedding_dim=D, qkv_dim=int(3 * qkv_dim)) 
    
    # #attention_values = self_attention(embeddings)

    # # # Multi-Head Attention Module     
    # class MultiHeadAttention(nn.Module):
    #     def __init__(self, embedding_dim, num_heads):
    #         super(MultiHeadAttention, self).__init__()

    #         self.num_heads = num_heads            
    #         self.embedding_dim = embedding_dim    # # embedding dimension 

    #         self.qkv_dim = embedding_dim // num_heads   # Dimension of key, query, and value can be calculated with embedding_dim and num_of_heads

    #         # initialise self-attention modules num_heads times
    #         self.multi_head_attention = nn.ModuleList([SelfAttention(embedding_dim, self.qkv_dim) for _ in range(num_heads)])

    #         # initialise weight matrix. 
    #         #self.W = nn.Parameter(torch.empty(1, num_heads * self.qkv_dim, embedding_dim).normal_(std=0.02))  
    #         self.W = nn.Parameter(torch.empty(1, num_heads * self.qkv_dim, embedding_dim).normal_(std=0.02)).cuda()

    #     def forward(self, x):
    #         # self-attention scores for each head
    #         attention_scores = [attention(x) for attention in self.multi_head_attention]

    #         # The outputs from all attention heads are concatenated and linearly transformed. 
    #         Z = torch.cat(attention_scores, -1)

    #         # This step ensures that the model can consider a comprehensive set of relationships captured by different heads.
    #         return torch.matmul(Z, self.W)

    # # initialise Multi-Head Attention object 
    # multi_head_attention = MultiHeadAttention(D, num_heads)

    # # calculate Multi-Head Attention score
    # #multi_head_attention_score = multi_head_attention(patch_embeddings_with_class_token)

    # class MLP(nn.Module):
    #     def __init__(self, embedding_dim, hidden_dim):
    #         super(MLP, self).__init__()

    #         # self.mlp = nn.Sequential(
    #         #                     nn.Linear(embedding_dim, hidden_dim),
    #         #                     nn.GELU(),
    #         #                     nn.Linear(hidden_dim, embedding_dim)
    #         #         )   
    #         self.mlp = nn.Sequential(
    #                             nn.Linear(embedding_dim, hidden_dim),
    #                             nn.GELU(),
    #                             nn.Linear(hidden_dim, embedding_dim)
    #                 ).cuda() 

    #     def forward(self, x):
    #         return self.mlp(x)

    # # # initialise MLP object                
    # #mlp = MLP(D, hidden_dim) 
    # hidden_dim = 3072
    # mlp = MLP(D, hidden_dim)
    
    # #output = mlp(multi_head_attention_score) 



    # # # Transformer Encoder Module       
    # class TransformerEncoder(nn.Module):
    #     def __init__(self, embedding_dim, num_heads, hidden_dim, dropout):
    #         super(TransformerEncoder, self).__init__()

    #         self.multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
    #         self.mlp = MLP(embedding_dim, hidden_dim)

    #         #self.layer_norm1 = nn.LayerNorm(embedding_dim)   
    #         self.layer_norm1 = nn.LayerNorm(embedding_dim).cuda()
    #         #self.layer_norm2 = nn.LayerNorm(embedding_dim) 
    #         self.layer_norm2 = nn.LayerNorm(embedding_dim).cuda()

    #         self.dropout1 = nn.Dropout(p=dropout)
    #         self.dropout2 = nn.Dropout(p=dropout)
    #         self.dropout3 = nn.Dropout(p=dropout)

    #     def forward(self, embeddings): 
    #         # # Applying dropout         
    #         dropout_embeddings = self.dropout1(embeddings) 
    #         # Layer normalization
    #         normalized_embeddings = self.layer_norm1(dropout_embeddings)
    #         #normalized_embeddings = self.layer_norm1(embeddings)
    #         # Calculation of multi-head attention
    #         attention_scores = self.multi_head_attention(normalized_embeddings)
    #         # Applying the second dropout
    #         dropout_attention_scores = self.dropout2(attention_scores)
    #         # Residual connection with second dropout output and initial input
    #         residuals_embeddings = embeddings + dropout_attention_scores
    #         #output = embeddings + attention_scores
    #         # apply layer normalization   
    #         normalized_residuals_embeddings = self.layer_norm2(residuals_embeddings)
    #         # aply MLP  
    #         transformed_results = self.mlp(normalized_residuals_embeddings)
    #         # Applying the third dropout
    #         dropout_transformed_results = self.dropout3(transformed_results)
    #         # # Residual connection with last dropout output and first residual output   
    #         output = residuals_embeddings + dropout_transformed_results 

    #         return output

    # # # init transformer encoder         
    # transformer_encoder = TransformerEncoder(embedding_dim=D, num_heads=num_heads, hidden_dim=hidden_dim, dropout=0.1) 

    # # # compute transformer encoder output
    # #output = transformer_encoder(embeddings)

    # class MLPHead(nn.Module):
    #     def __init__(self, embedding_dim, num_classes, is_train=True):
    #         super(MLPHead, self).__init__()
    #         self.num_classes = num_classes
    #         # this part is taken from torchvision implementation  
    #         if is_train:
    #             self.head = nn.Sequential(
    #                                     nn.Linear(embedding_dim, 3072),  # hidden layer
    #                                     nn.Tanh(),
    #                                     nn.Linear(3072, num_classes)    # output layer
    #                             ).cuda()
    #         else:
    #             # single linear layer 
    #             self.head = nn.Linear(embedding_dim, num_classes).cuda() 

    #     def forward(self, x):
    #         return self.head(x)
    
    # class PixelMLPHead(nn.Module):
    #     def __init__(self, in_channels=768, num_classes=10):
    #         super(PixelMLPHead, self).__init__()
            
    #         # Fully connected layers to process the input   
    #         self.fc1 = nn.Linear(in_channels, 1024)  # Reduce to 1024 channels
    #         self.fc2 = nn.Linear(1024, 128 * 128)    # Output size should match the target 128x128 grid

    #         # Final convolution layer to produce segmentation map
    #         self.seg_head = nn.Conv2d(1, num_classes, kernel_size=1, stride=1)

    #     def forward(self, x):
    #         # x: (batch_size, in_channels) -> (100, 768) for each batch element  
            
    #         # Apply fully connected layers   
    #         x = F.relu(self.fc1(x)) 
    #         x = F.relu(self.fc2(x))

    #         # Reshape the output to (batch_size, 1, 128, 128)
    #         batch_size = x.size(0)
    #         x = x.view(batch_size, 1, 128, 128)  # Now (batch_size, 1, 128, 128)

    #         # Apply the segmentation head to get class scores
    #         x = self.seg_head(x)  # Output shape: (batch_size, num_classes, 128, 128)

    #         return x

    # class ViTSegmentationHead(nn.Module):
    #     def __init__(self, in_channels=768, num_classes=21):
    #         super(ViTSegmentationHead, self).__init__()

    #         # First convolutional layer to process the input features
    #         self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1)
    #         self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
    #         self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
    #         self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)

    #         # Upsampling layers to restore spatial resolution
    #         self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4)  # Upsample by 4x
    #         self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4)
    #         self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=4)
    #         self.upconv4 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=4)

    #         # Final convolution layer for the segmentation map (1x1)
    #         self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)

    #     def forward(self, x):
    #         # x: (batch_size, num_patches, patch_embedding_dim), i.e., (batch_size, 100, 768)
            
    #         # Reshape the ViT output to (batch_size, in_channels, height, width)
    #         # Assuming we have a grid (10x10) of patches, let's reshape the input (batch_size, 768) -> (batch_size, 768, 1, 1)
    #         batch_size = x.size(0)
    #         x = x.view(batch_size, 768, 1, 1)  # Reshaping to (batch_size, 768, 1, 1)
            
    #         # Apply convolutions
    #         x = F.relu(self.conv1(x))  # (batch_size, 512, 1, 1)
    #         x = F.relu(self.conv2(x))  # (batch_size, 256, 1, 1)
    #         x = F.relu(self.conv3(x))  # (batch_size, 128, 1, 1)
    #         x = F.relu(self.conv4(x))  # (batch_size, 64, 1, 1)
            
    #         # Upsample the feature map to the target size (128x128)
    #         x = self.upconv1(x)  # (batch_size, 64, 4, 4)
    #         x = self.upconv2(x)  # (batch_size, 128, 16, 16)
    #         x = self.upconv3(x)  # (batch_size, 256, 64, 64)
    #         x = self.upconv4(x)  # (batch_size, 512, 256, 256)
            
    #         # Apply final segmentation head to get class scores 
    #         x = self.seg_head(x)  # Output shape: (batch_size, num_classes, 128, 128)

    #         return x
    
    # # # Classifier "token" as used by standard language architectures        
    # #class_token_output = output[:, 0]     

    # # # initialise number of classes  
    # #n_class = 10
    # n_class = 11

    # # # initialise classification head   
    # mlp_head = MLPHead(D, n_class) 

    # #cls_output = mlp_head(class_token_output)



    # # # VisionTranformer Module     
    # class VisionTransformer(nn.Module):
    #     def __init__(self, patch_size=16, image_size=224, C=3,
    #                     num_layers=12, embedding_dim=768, num_heads=12, hidden_dim=3072,
    #                             dropout_prob=0.1, num_classes=10):
    #         super(VisionTransformer, self).__init__()

    #         self.patch_size = patch_size
    #         self.C = C

    #         # get the number of patches of the image
    #         self.num_patches = int(image_size ** 2 / patch_size ** 2) # (width * height) / (patch_size**2)

    #         # trainable linear projection for mapping dimension of patches (weight matrix E)
    #         self.W = nn.Parameter(torch.empty(1, patch_size * patch_size * C, embedding_dim).normal_(std=0.02))

    #         # position embeddings
    #         self.positional_embeddings = nn.Parameter(torch.empty(1, self.num_patches + 1, embedding_dim).normal_(std=0.02))

    #         # learnable class tokens
    #         self.class_tokens = nn.Parameter(torch.rand(1, D))

    #         # transformer encoder
    #         self.transformer_encoder = nn.Sequential(*[
    #             TransformerEncoder(embedding_dim, num_heads, hidden_dim, dropout_prob) for _ in range(num_layers)
    #         ])

    #         # # mlp head     
    #         #self.mlp_head = MLPHead(embedding_dim, num_classes)  
    #         self.mlp_head = PixelMLPHead(embedding_dim, num_classes)
    #         #self.mlp_head = ViTSegmentationHead(embedding_dim, num_classes)
            
    #     def forward(self, images): 
    #         # # get patch size and channel size         
    #         P, C = self.patch_size, self.C 

    #         # create image patches
    #         patches = images.unfold(1, C, C).unfold(2, P, P).unfold(3, P, P).contiguous().view(images.size(0), -1, C * P * P).float()

    #         # patch embeddings
    #         patch_embeddings = torch.matmul(patches , self.W)

    #         # # class token + patch_embeddings            
    #         batch_class_token = self.class_tokens.expand(patch_embeddings.shape[0], -1, -1) 
    #         patch_embeddings_with_class_token = torch.cat([batch_class_token, patch_embeddings], dim=1)

    #         # add positional embedding
    #         embeddings = patch_embeddings_with_class_token + self.positional_embeddings

    #         # execute Transformer encoders  
    #         transformer_encoder_output = self.transformer_encoder(embeddings)

    #         # Classifier "token" as used by standard language architectures 
    #         output_class_token = transformer_encoder_output[:, 0]

    #         return self.mlp_head(output_class_token)

    # # # init vision transformer model                                         
    # #num_layers = 12     
    # num_layers = 5     
    # # vision_transformer = VisionTransformer(patch_size=P, 
    # #                                     image_size=desired_image_size, 
    # #                                     C=C,
    # #                                     num_layers=num_layers, 
    # #                                     embedding_dim=D, 
    # #                                     num_heads=num_heads, 
    # #                                     hidden_dim=hidden_dim, 
    # #                                     dropout_prob=0.1,   
    # #                                     num_classes=n_class).cuda()       

    # model = VisionTransformer(patch_size=P, 
    #                                     image_size=desired_image_size, 
    #                                     C=C,
    #                                     num_layers=num_layers, 
    #                                     embedding_dim=D, 
    #                                     num_heads=num_heads, 
    #                                     hidden_dim=hidden_dim, 
    #                                     dropout_prob=0.1, 
    #                                     num_classes=n_class).cuda()  
    
    # model.train()                                                              



    # from mmcv.cnn import ConvModule      
    # from timm.models.vision_transformer import PatchEmbed, Block  
    # from mmseg.models.decode_heads.decode_head import BaseDecodeHead   
    # from mmseg.models.decode_heads.psp_head import PPM 
    # from mmseg.models.utils.wrappers import resize 
    
    # class UPerHead(BaseDecodeHead):
    #     """Unified Perceptual Parsing for Scene Understanding            

    #     This head is the implementation of `UPerNet
    #     <https://arxiv.org/abs/1807.10221>`_.

    #     Args:
    #         pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
    #             Module applied on the last feature. Default: (1, 2, 3, 6).
    #     """

    #     def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
    #         super().__init__(input_transform='multiple_select', **kwargs)
    #         # PSP Module
    #         self.psp_modules = PPM(
    #             pool_scales,
    #             self.in_channels[-1],
    #             self.channels,
    #             conv_cfg=self.conv_cfg,
    #             norm_cfg=self.norm_cfg,
    #             act_cfg=self.act_cfg,
    #             align_corners=self.align_corners)
    #         self.bottleneck = ConvModule(
    #             self.in_channels[-1] + len(pool_scales) * self.channels,
    #             self.channels,
    #             3,
    #             padding=1,
    #             conv_cfg=self.conv_cfg,
    #             norm_cfg=self.norm_cfg,
    #             act_cfg=self.act_cfg)
    #         # FPN Module
    #         self.lateral_convs = nn.ModuleList()
    #         self.fpn_convs = nn.ModuleList()
    #         for in_channels in self.in_channels[:-1]:  # skip the top layer
    #             l_conv = ConvModule(
    #                 in_channels,
    #                 self.channels,
    #                 1,
    #                 conv_cfg=self.conv_cfg,
    #                 norm_cfg=self.norm_cfg,
    #                 act_cfg=self.act_cfg,
    #                 inplace=False)
    #             fpn_conv = ConvModule(
    #                 self.channels,
    #                 self.channels,
    #                 3,
    #                 padding=1,
    #                 conv_cfg=self.conv_cfg,
    #                 norm_cfg=self.norm_cfg,
    #                 act_cfg=self.act_cfg,
    #                 inplace=False)
    #             self.lateral_convs.append(l_conv)
    #             self.fpn_convs.append(fpn_conv)

    #         self.fpn_bottleneck = ConvModule(
    #             len(self.in_channels) * self.channels,
    #             self.channels,
    #             3,
    #             padding=1,
    #             conv_cfg=self.conv_cfg,
    #             norm_cfg=self.norm_cfg,
    #             act_cfg=self.act_cfg)

    #     def psp_forward(self, inputs):
    #         """Forward function of PSP module."""
    #         x = inputs[-1]
    #         psp_outs = [x]
    #         psp_outs.extend(self.psp_modules(x))
    #         psp_outs = torch.cat(psp_outs, dim=1)
    #         output = self.bottleneck(psp_outs)

    #         return output

    #     def _forward_feature(self, inputs):
    #         """Forward function for feature maps before classifying each pixel with
    #         ``self.cls_seg`` fc.

    #         Args:
    #             inputs (list[Tensor]): List of multi-level img features.

    #         Returns:
    #             feats (Tensor): A tensor of shape (batch_size, self.channels,
    #                 H, W) which is feature map for last layer of decoder head.
    #         """
    #         inputs = self._transform_inputs(inputs)

    #         # build laterals
    #         laterals = [
    #             lateral_conv(inputs[i])
    #             for i, lateral_conv in enumerate(self.lateral_convs)
    #         ]

    #         laterals.append(self.psp_forward(inputs))

    #         #print(laterals[i - 1].shape[2:])
    #         #sadfasdf

    #         # build top-down path
    #         used_backbone_levels = len(laterals)
    #         for i in range(used_backbone_levels - 1, 0, -1):
    #             prev_shape = laterals[i - 1].shape[2:]
    #             laterals[i - 1] = laterals[i - 1] + resize(
    #                 laterals[i],
    #                 size=prev_shape,
    #                 mode='bilinear',
    #                 align_corners=self.align_corners)

    #         # build outputs
    #         fpn_outs = [
    #             self.fpn_convs[i](laterals[i])
    #             for i in range(used_backbone_levels - 1)
    #         ]
    #         # append psp feature
    #         fpn_outs.append(laterals[-1])

    #         for i in range(used_backbone_levels - 1, 0, -1):
    #             fpn_outs[i] = resize(
    #                 fpn_outs[i],
    #                 size=fpn_outs[0].shape[2:],
    #                 mode='bilinear',
    #                 align_corners=self.align_corners)
    #         fpn_outs = torch.cat(fpn_outs, dim=1)
    #         feats = self.fpn_bottleneck(fpn_outs)
    #         return feats

    #     def forward(self, inputs):
    #         """Forward function."""
    #         output = self._forward_feature(inputs)
    #         output = self.cls_seg(output)
    #         return output

    # class ViTEncoder(nn.Module):
    #     """ 
    #         VisionTransformer backbone    
    #     """
    #     def __init__(self, chw:tuple=(10, 128, 128), patch_size:int=4, output_dim:int=10,
    #                 embed_dim=768, depth=12, num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm, 
    #                 ):
    #         super().__init__()

    #         # Attributes
    #         self.chw = chw  # (C, H, W)
    #         self.in_c = chw[0]
    #         self.img_size = chw[1]
    #         self.patch_size = patch_size
    #         self.output_dim = output_dim
    #         self.embed_dim = embed_dim
    #         self.depth = depth
    #         self.num_heads = num_heads
    #         self.mlp_ratio = mlp_ratio
    #         self.norm_layer = norm_layer
            
    #         # --------------------------------------------------------------------------
    #         # MAE encoder specifics     
    #         self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.in_c, self.embed_dim)
    #         num_patches = self.patch_embed.num_patches

    #         self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
    #         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
    #                                     requires_grad=False)  # # learnable with sin-cos embedding init      

    #         self.blocks = nn.ModuleList([
    #             Block(self.embed_dim, self.num_heads, self.mlp_ratio, qkv_bias=True, norm_layer= self.norm_layer)
    #             for i in range(self.depth)])
    #         self.norm = self.norm_layer(self.embed_dim)

    #         self.initialize_weights()
    #         # --------------------------------------------------------------------------

    #     def initialize_weights(self):
    #         # initialization
    #         # initialize (and freeze) pos_embed by sin-cos embedding
    #         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
    #                                             cls_token=True)
    #         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    #         # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
    #         w = self.patch_embed.proj.weight.data
    #         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    #         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
    #         torch.nn.init.normal_(self.cls_token, std=.02)

    #         # initialize nn.Linear and nn.LayerNorm
    #         self.apply(self._init_weights)

    #     def _init_weights(self, m):
    #         if isinstance(m, nn.Linear):
    #             # we use xavier_uniform following official JAX ViT:
    #             torch.nn.init.xavier_uniform_(m.weight)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)

    #     def forward(self, x):
    #         # embed patches
    #         # B, C, H, W = x.shape
    #         x = self.patch_embed(x)

    #         # add pos embed w/o cls token
    #         x = x + self.pos_embed[:, 1:, :]

    #         # append cls token
    #         cls_token = self.cls_token + self.pos_embed[:, :1, :]
    #         cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    #         x = torch.cat((cls_tokens, x), dim=1)

    #         # apply Transformer blocks
    #         hidden_states = []
    #         for blk in self.blocks:
    #             x = blk(x)
    #             hidden_states.append(x)
    #         x = self.norm(x)
    #         hidden_states[-1] = x
    #         # # remove cls token
    #         #x = x[:, 1:, :]

    #         return x, hidden_states

    # class ViTUperNet(nn.Module):
    #     """ 
    #     ViT backbone with UperNet decoder head   
    #     """

    #     def __init__(self,  chw:tuple=(10, 128, 128), patch_size:int=4, output_dim:int=11,
    #                 embed_dim=768, depth=12, num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm, 
    #                 decoder_out_channels = 256, decoder_in_index = [2,5,8,11], decoder_pool_scales = (1,2,3,6), decoder_norm= {'type': 'BN2d'}):
    #         super().__init__()

    #         # Attributes
    #         self.chw = chw  # (C, H, W)
    #         self.in_c = chw[0]
    #         self.img_size = chw[1]
    #         self.patch_size = patch_size
    #         self.num_classes = output_dim
    #         self.embed_dim = embed_dim
    #         self.depth = depth
    #         self.num_heads = num_heads
    #         self.mlp_ratio = mlp_ratio
    #         self.norm_layer = norm_layer
    #         self.decoder_in_index = decoder_in_index
    #         self.decoder_out_channels = decoder_out_channels
    #         self.decoder_pool_scales = decoder_pool_scales
    #         self.decoder_norm = decoder_norm
    #         assert all(element < self.depth for element in self.decoder_in_index), f"Please select intermediate features from one of the {self.depth} layers"
        
    #         # --------------------------------------------------------------------------
    #         # # encoder specifics      
    #         self.vit_encoder = ViTEncoder(chw=self.chw, 
    #                                     patch_size=self.patch_size, output_dim=self.num_classes,
    #                                     embed_dim=self.embed_dim, depth=self.depth, num_heads=self.num_heads,
    #                                     mlp_ratio=self.mlp_ratio, norm_layer=self.norm_layer)
    
    #         # --------------------------------------------------------------------------

    #         # --------------------------------------------------------------------------
    #         # decoder UperNet   
    #         # upsample/downsample the input before feeding it to UperNet
    #         self.fpn1 = nn.Sequential(nn.ConvTranspose2d(in_channels = self.embed_dim, out_channels = self.embed_dim//2, kernel_size= 2, stride= 2),
    #                                 nn.BatchNorm2d(self.embed_dim//2), nn.ReLU(),
    #                                 nn.ConvTranspose2d(in_channels=self.embed_dim//2, out_channels= self.embed_dim//4, kernel_size= 2, stride= 2))
    #         self.fpn2 = nn.Sequential(nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels= self.embed_dim//2, kernel_size= 2, stride = 2),
    #                                 nn.BatchNorm2d(self.embed_dim//2), nn.ReLU())
             
    #         self.fpn3 = nn.Identity()  
    #         self.fpn4 = nn.MaxPool2d(kernel_size= 2, stride = 2) 
            
    #         self.sample_list_base = nn.ModuleList([self.fpn1, self.fpn2, self.fpn3, self.fpn4])
    #         self.decoder_upernet = UPerHead(in_channels =[self.embed_dim//4, self.embed_dim//2, self.embed_dim, self.embed_dim] , channels = self.decoder_out_channels, 
    #                                 num_classes = self.num_classes, norm_cfg = self.decoder_norm, in_index = self.decoder_in_index)
 
    #         self.learnable_upsample = nn.Sequential(
    #             nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=4, stride=4, padding=0),
    #             nn.BatchNorm2d(self.num_classes),
    #             nn.ReLU(inplace=True)
    #         )  
 
    #         # --------------------------------------------------------------------------  
        
    #     def reshape_vit_features(self, input):
    #         B,N,D = input.shape
    #         # B = batch_size , N = number of patches, D = embedding dimension 
    #         # Reshape to obtain spatial resolutions, i.e. (B, N, D) -> (B, H/P, W/P, D)
    #         H_p = self.img_size // self.patch_size
    #         W_p = self.img_size// self.patch_size
    #         input = input.view(B, H_p, W_p, D)
    #         # Permute to (B, D, H/P, W/P), i.e. needed for UPerNet
    #         input = input.permute(0, 3, 1, 2)
    #         return input
            

    #     def forward(self, x):
    #         # B, N, D = hidden_states[i].shape       
    #         _, hidden_states = self.vit_encoder(x)
            
    #         #print(hidden_states.shape)  
    #         #sadfasdf
            
    #         # select desired intermediate features: remove cls token + reshape to appropriate size + upsample/downsample + extract their dimensions   
    #         for i, sample in zip(self.decoder_in_index, self.sample_list_base):
    #             hidden_states[i] = sample(self.reshape_vit_features(hidden_states[i][:,1:, :]))
    #         # decoder
    #         outputs = self.decoder_upernet(hidden_states)
            
    #         outputs = self.learnable_upsample(outputs)  
            
    #         return outputs
        
    # def vit_upernet_large(**kwargs):
    #     model = ViTUperNet(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, decoder_in_index=[5,11,17,23],
    #                 norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    #     return model

    # def vit_upernet_pretrained(checkpoint, chw:tuple=(10, 128, 128), patch_size:int=4, output_dim:int=11, freeze_body = True, **kwargs):
    #     # # load pre-trained model weights       
    #     model = vit_upernet_large(chw = chw, patch_size = patch_size, output_dim = output_dim, **kwargs) 
    #     msg = model.vit_encoder.load_state_dict(checkpoint, strict= False)
    #     print(msg)
        
    #     if freeze_body:
    #         for _, param in model.vit_encoder.named_parameters():
    #                 param.requires_grad = False

    #     return model 
    
    # # model = ViTUperNet(chw:tuple=(10, 128, 128), patch_size:int=4, output_dim:int=11,
    # #                 embed_dim=768, depth=12, num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm, 
    # #                 decoder_out_channels = 256, decoder_in_index = [2,5,8,11], decoder_pool_scales = (1,2,3,6), decoder_norm= {'type': 'BN2d'}))       
    
    # #model = ViTUperNet()                      
    # model = ViTUperNet(patch_size=16)        
    
    # model.train()     



    # # from mmcv.cnn import ConvModule 
    # # from mmseg.models.utils.wrappers import resize
    # # from mmseg.models.decode_heads.decode_head import BaseDecodeHead
    # # from mmseg.models.decode_heads.psp_head import PPM


    # # class UPerHead(BaseDecodeHead):
    # #     """Unified Perceptual Parsing for Scene Understanding.

    # #     This head is the implementation of `UPerNet
    # #     <https://arxiv.org/abs/1807.10221>`_.

    # #     Args:
    # #         pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
    # #             Module applied on the last feature. Default: (1, 2, 3, 6).
    # #     """

    # #     def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
    # #         super().__init__(input_transform='multiple_select', **kwargs)
    # #         # PSP Module
    # #         self.psp_modules = PPM(
    # #             pool_scales,
    # #             self.in_channels[-1],
    # #             self.channels,
    # #             conv_cfg=self.conv_cfg,
    # #             norm_cfg=self.norm_cfg,
    # #             act_cfg=self.act_cfg,
    # #             align_corners=self.align_corners)
    # #         self.bottleneck = ConvModule(
    # #             self.in_channels[-1] + len(pool_scales) * self.channels,
    # #             self.channels,
    # #             3,
    # #             padding=1,
    # #             conv_cfg=self.conv_cfg,
    # #             norm_cfg=self.norm_cfg,
    # #             act_cfg=self.act_cfg)
    # #         # FPN Module
    # #         self.lateral_convs = nn.ModuleList()
    # #         self.fpn_convs = nn.ModuleList()
    # #         for in_channels in self.in_channels[:-1]:  # skip the top layer
    # #             l_conv = ConvModule(
    # #                 in_channels,
    # #                 self.channels,
    # #                 1,
    # #                 conv_cfg=self.conv_cfg,
    # #                 norm_cfg=self.norm_cfg,
    # #                 act_cfg=self.act_cfg,
    # #                 inplace=False)
    # #             fpn_conv = ConvModule(
    # #                 self.channels,
    # #                 self.channels,
    # #                 3,
    # #                 padding=1,
    # #                 conv_cfg=self.conv_cfg,
    # #                 norm_cfg=self.norm_cfg,
    # #                 act_cfg=self.act_cfg,
    # #                 inplace=False)
    # #             self.lateral_convs.append(l_conv)
    # #             self.fpn_convs.append(fpn_conv)

    # #         self.fpn_bottleneck = ConvModule(
    # #             len(self.in_channels) * self.channels,
    # #             self.channels,
    # #             3,
    # #             padding=1,
    # #             conv_cfg=self.conv_cfg,
    # #             norm_cfg=self.norm_cfg,
    # #             act_cfg=self.act_cfg)

    # #     def psp_forward(self, inputs):
    # #         """Forward function of PSP module."""
    # #         x = inputs[-1]
    # #         psp_outs = [x]
    # #         psp_outs.extend(self.psp_modules(x))
    # #         psp_outs = torch.cat(psp_outs, dim=1)
    # #         output = self.bottleneck(psp_outs)

    # #         return output

    # #     def _forward_feature(self, inputs):
    # #         """Forward function for feature maps before classifying each pixel with
    # #         ``self.cls_seg`` fc.

    # #         Args:
    # #             inputs (list[Tensor]): List of multi-level img features.

    # #         Returns:
    # #             feats (Tensor): A tensor of shape (batch_size, self.channels,
    # #                 H, W) which is feature map for last layer of decoder head.
    # #         """
    # #         inputs = self._transform_inputs(inputs)

    # #         # build laterals
    # #         laterals = [
    # #             lateral_conv(inputs[i])
    # #             for i, lateral_conv in enumerate(self.lateral_convs)
    # #         ]

    # #         laterals.append(self.psp_forward(inputs))

    # #         # build top-down path
    # #         used_backbone_levels = len(laterals)
    # #         for i in range(used_backbone_levels - 1, 0, -1):
    # #             prev_shape = laterals[i - 1].shape[2:]
    # #             laterals[i - 1] = laterals[i - 1] + resize(
    # #                 laterals[i],
    # #                 size=prev_shape,
    # #                 mode='bilinear',
    # #                 align_corners=self.align_corners)

    # #         # build outputs
    # #         fpn_outs = [
    # #             self.fpn_convs[i](laterals[i])
    # #             for i in range(used_backbone_levels - 1)
    # #         ]
    # #         # append psp feature
    # #         fpn_outs.append(laterals[-1])

    # #         for i in range(used_backbone_levels - 1, 0, -1):
    # #             fpn_outs[i] = resize(
    # #                 fpn_outs[i],
    # #                 size=fpn_outs[0].shape[2:],
    # #                 mode='bilinear',
    # #                 align_corners=self.align_corners)
    # #         fpn_outs = torch.cat(fpn_outs, dim=1)
    # #         feats = self.fpn_bottleneck(fpn_outs)
    # #         return feats

    # #     def forward(self, inputs):
    # #         """Forward function."""
    # #         output = self._forward_feature(inputs)
    # #         output = self.cls_seg(output)
    # #         return output

    # # class ScaleSkip2D(nn.Module):
    # #     """
    # #     Learnable channel-wise scale and bias for skip connections.

    # #     Parameters
    # #     ----------
    # #     channels : int
    # #         Number of channels in the input

    # #     drop_y : float
    # #         Probability of dropping a channel in the skip connection.
    # #         Drops are replaces with Gaussian noise.

    # #     signal_to_noise : tuple or None
    # #         Range of signal to noise ratios to use for the dropped channels. 0.0 is pure noise, 1.0 is pure signal.
    # #         The amount of signal is randomly sampled from this range for each channel.
    # #         If None, no signal is added to the dropped channels.
    # #         default: (0.1, 0.9)

    # #     size : float
    # #         Standard deviation of the normal distribution to sample initial values from.
    # #         default: 0.01
    # #     """

    # #     def __init__(self, channels, drop_y=None, signal_to_noise=(0.1, 0.9), size=0.01):
    # #         super(ScaleSkip2D, self).__init__()
    # #         self.channels = channels
    # #         self.drop_y = drop_y
    # #         self.size = size

    # #         # Learnable scale and bias
    # #         self.x_skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1)) 
    # #         self.y_skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1))
    # #         self.y_skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
    # #         self.x_skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))

    # #         if self.drop_y is not None and self.drop_y > 0.0:
    # #             self.drop_y = GaussianDropout2d(self.drop_y, signal_to_noise=signal_to_noise)
    # #         else:
    # #             self.drop_y = None

    # #         self.set_weights()
    # #         while torch.any(self.x_skipscale == 0) or torch.any(self.y_skipscale == 0) or torch.any(
    # #                 self.y_skipbias == 0
    # #         ) or torch.any(self.x_skipbias == 0):
    # #             self.set_weights()

    # #     def set_weights(self):
    # #         nn.init.trunc_normal_(self.x_skipscale, 1.0, self.size)
    # #         nn.init.trunc_normal_(self.y_skipscale, 1.0, self.size)
    # #         nn.init.trunc_normal_(self.y_skipbias, 0.0, self.size)
    # #         nn.init.trunc_normal_(self.x_skipbias, 0.0, self.size)

    # #     def forward(self, x, y):
    # #         x = (x * self.x_skipscale) + self.x_skipbias
    # #         y = (y * self.y_skipscale) + self.y_skipbias

    # #         if self.drop_y is not None:
    # #             y = self.drop_y(y)

    # #         return x + y


    # # class ScaleSkip1D(nn.Module):
    # #     """ Learnable weight and bias for 1D skip connections. """

    # #     def __init__(self, drop_y=None, size=0.01):
    # #         super(ScaleSkip1D, self).__init__()

    # #         self.size = size
    # #         self.drop_y = drop_y

    # #         # Learnable scale and bias
    # #         self.x_skipscale = nn.Parameter(torch.ones(1, 1))
    # #         self.y_skipscale = nn.Parameter(torch.ones(1, 1))
    # #         self.y_skipbias = nn.Parameter(torch.zeros(1, 1))
    # #         self.x_skipbias = nn.Parameter(torch.zeros(1, 1))

    # #         self.set_weights()
    # #         while torch.any(self.x_skipscale == 0) or torch.any(self.y_skipscale == 0) or torch.any(
    # #                 self.y_skipbias == 0
    # #         ) or torch.any(self.x_skipbias == 0):
    # #             self.set_weights()

    # #         if self.drop_y is not None and self.drop_y > 0.0:
    # #             self.drop_y = GaussianDropout1d(self.drop_y)
    # #         else:
    # #             self.drop_y = None

    # #     def set_weights(self):
    # #         nn.init.trunc_normal_(self.x_skipscale, 1.0, self.size)
    # #         nn.init.trunc_normal_(self.y_skipscale, 1.0, self.size)
    # #         nn.init.trunc_normal_(self.y_skipbias, 0.0, self.size)
    # #         nn.init.trunc_normal_(self.x_skipbias, 0.0, self.size)

    # #     def forward(self, x, y):
    # #         x = (x * self.x_skipscale) + self.x_skipbias
    # #         y = (y * self.y_skipscale) + self.y_skipbias

    # #         if self.drop_y is not None:
    # #             y = self.drop_y(y)

    # #         return x + y

    # # class SE_Block(nn.Module):
    # #     """ credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4 """

    # #     def __init__(self, channels, reduction=16):
    # #         super().__init__()
    # #         self.reduction = reduction
    # #         self.squeeze = nn.AdaptiveAvgPool2d(1)
    # #         self.excitation = nn.Sequential(
    # #             nn.Linear(channels, max(1, channels // self.reduction), bias=False),
    # #             nn.GELU(),
    # #             nn.Linear(max(1, channels // self.reduction), channels, bias=False),
    # #             nn.Sigmoid()
    # #         )

    # #     def forward(self, x):
    # #         bs, c, _, _ = x.shape
    # #         y = self.squeeze(x).view(bs, c)
    # #         y = self.excitation(y).view(bs, c, 1, 1)

    # #         return x * y.expand_as(x)

    # # class CNNBlock(nn.Module):
    # #     """
    # #     This is a standard CNN block with a 1x1 convolutional matcher for the skip connection.
    # #     It adds a learnable scale and bias to the skip connection.

    # #     Parameters
    # #     ----------
    # #     channels_in : int
    # #         Number of channels in the input

    # #     channels_out : int or None
    # #         Number of channels in the output. If None, the number of channels is unchanged.
    # #         default: None

    # #     group_size : int
    # #         Number of groups for the 3x3 convolution.
    # #         default: 1

    # #     activation : torch.nn.Module
    # #         Activation function to use after the first convolution.
    # #         default: torch.nn.GELU()

    # #     activation_out : torch.nn.Module or None
    # #         Activation function to use after the last convolution.
    # #         If None, the same activation as the first convolution is used.
    # #         default: None

    # #     chw : tuple or None
    # #         Height and width of the input. If None, batch norm is used instead of layer norm.
    # #         default: None
    # #     """

    # #     def __init__(
    # #             self,
    # #             channels_in,
    # #             channels_out=None,
    # #             chw=None,
    # #             group_size=1,
    # #             activation=nn.GELU(),
    # #             activation_out=None,
    # #             residual=True,
    # #             reduction=1,
    # #     ):
    # #         super().__init__()

    # #         assert chw is not None, "chw must be specified"

    # #         self.channels_in = channels_in
    # #         self.channels_out = channels_in if channels_out is None else channels_out
    # #         self.channels_internal = self.channels_out // reduction
    # #         self.chw = chw
    # #         self.group_size = group_size
    # #         self.activation = activation
    # #         self.activation_out = activation if activation_out is None else activation_out
    # #         self.residual = residual
    # #         self.reduction = reduction
    # #         self.squeeze = SE_Block(self.channels_out, 16)

    # #         self.matcher = nn.Conv2d(
    # #             self.channels_in, self.channels_out, 1, padding=0,
    # #             bias=False
    # #             ) if self.channels_in != self.channels_out else None

    # #         self.norm1 = nn.LayerNorm([self.channels_internal, self.chw[1], self.chw[2]])
    # #         self.norm2 = nn.LayerNorm([self.channels_internal, self.chw[1], self.chw[2]])

    # #         self.conv1 = nn.Conv2d(self.channels_in, self.channels_internal, 1, padding=0, bias=False)
    # #         self.conv2 = nn.Conv2d(
    # #             self.channels_internal, self.channels_internal, 3, padding=1, groups=self.group_size,
    # #             bias=False, padding_mode="replicate"
    # #             )
    # #         self.conv3 = nn.Conv2d(self.channels_internal, self.channels_out, 1, padding=0, bias=True)

    # #         self.scaler = ScaleSkip2D(self.channels_out) if self.residual else None

    # #     def forward(self, x):
    # #         identity = x if self.matcher is None else self.matcher(x)

    # #         x = self.conv1(x)
    # #         x = self.norm1(x)
    # #         x = self.activation(x)

    # #         x = self.conv2(x)
    # #         x = self.norm2(x)
    # #         x = self.activation(x)

    # #         x = self.conv3(x)
    # #         x = self.squeeze(x)

    # #         if self.residual:
    # #             x = self.scaler(x, identity)

    # #         x = self.activation_out(x)
    # #         return x

    # # from timm.models.vision_transformer import PatchEmbed, Block 
    # # class ViTEncoder(nn.Module):
    # #     """ 
    # #         VisionTransformer backbone   
    # #     """
    # #     def __init__(self, chw:tuple=(10, 128, 128), patch_size:int=4, output_dim:int=10,
    # #                 embed_dim=768, depth=12, num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm, 
    # #                 ):
    # #         super().__init__() 

    # #         # Attributes
    # #         self.chw = chw  # (C, H, W)
    # #         self.in_c = chw[0]
    # #         self.img_size = chw[1]
    # #         self.patch_size = patch_size
    # #         self.output_dim = output_dim
    # #         self.embed_dim = embed_dim
    # #         self.depth = depth
    # #         self.num_heads = num_heads
    # #         self.mlp_ratio = mlp_ratio
    # #         self.norm_layer = norm_layer
            

    # #         # --------------------------------------------------------------------------
    # #         # MAE encoder specifics
    # #         self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.in_c, self.embed_dim)
    # #         num_patches = self.patch_embed.num_patches

    # #         self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
    # #         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
    # #                                     requires_grad=False)  # learnable with sin-cos embedding init

    # #         self.blocks = nn.ModuleList([
    # #             Block(self.embed_dim, self.num_heads, self.mlp_ratio, qkv_bias=True, norm_layer= self.norm_layer)
    # #             for i in range(self.depth)])
    # #         self.norm = self.norm_layer(self.embed_dim)

            
    # #         self.initialize_weights()
    # #         # --------------------------------------------------------------------------

    # #     def initialize_weights(self):
    # #         # initialization
    # #         # initialize (and freeze) pos_embed by sin-cos embedding
    # #         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
    # #                                             cls_token=True)
    # #         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    # #         # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
    # #         w = self.patch_embed.proj.weight.data
    # #         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    # #         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
    # #         torch.nn.init.normal_(self.cls_token, std=.02)

    # #         # initialize nn.Linear and nn.LayerNorm
    # #         self.apply(self._init_weights)

    # #     def _init_weights(self, m):
    # #         if isinstance(m, nn.Linear):
    # #             # we use xavier_uniform following official JAX ViT:
    # #             torch.nn.init.xavier_uniform_(m.weight)
    # #             if isinstance(m, nn.Linear) and m.bias is not None:
    # #                 nn.init.constant_(m.bias, 0)
    # #         elif isinstance(m, nn.LayerNorm):
    # #             nn.init.constant_(m.bias, 0)
    # #             nn.init.constant_(m.weight, 1.0)


    # #     def forward(self, x):
    # #         # embed patches
    # #         # B, C, H, W = x.shape
    # #         x = self.patch_embed(x)

    # #         # add pos embed w/o cls token
    # #         x = x + self.pos_embed[:, 1:, :]

    # #         # append cls token
    # #         cls_token = self.cls_token + self.pos_embed[:, :1, :]
    # #         cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    # #         x = torch.cat((cls_tokens, x), dim=1)

    # #         # apply Transformer blocks
    # #         hidden_states = []
    # #         for blk in self.blocks:
    # #             x = blk(x)
    # #             hidden_states.append(x)
    # #         x = self.norm(x)
    # #         hidden_states[-1] = x
    # #         # # remove cls token
    # #         #x = x[:, 1:, :]

    # #         return x, hidden_states



    # # class FoundationViTEncoder(nn.Module):
    # #     def __init__(
    # #             self,
    # #             chw=(3, 64, 64),  # Default image size
    # #             patch_size=4,
    # #             embed_dim=768,
    # #             depth=12,
    # #             num_heads=16,
    # #             mlp_ratio=4,
    # #             norm_layer=nn.LayerNorm,
    # #             latent_dim=512
    # #     ):
    # #         super().__init__()
            
    # #         self.vit_encoder = ViTEncoder(
    # #             chw=chw,
    # #             patch_size=patch_size,
    # #             embed_dim=embed_dim,
    # #             depth=depth,
    # #             num_heads=num_heads,
    # #             mlp_ratio=mlp_ratio,
    # #             norm_layer=norm_layer
    # #         )
            
    # #         self.latent_dim = latent_dim
    # #         self.linear_proj = nn.Linear(embed_dim, latent_dim)
            
    # #         self.head_clouds = nn.Linear(latent_dim, 4)
    # #         self.head_landcover = nn.Linear(latent_dim, 11)
    # #         self.head_buildings = nn.Sequential(
    # #             nn.Linear(latent_dim, 1),
    # #             nn.Sigmoid()
    # #         )
    # #         self.head_coords = nn.Sequential(
    # #             nn.Linear(latent_dim, 4),
    # #             nn.Sigmoid()
    # #         )
        
    # #     def forward(self, x):
    # #         vit_output, hidden_states = self.vit_encoder(x)  # Extract ViT embeddings
    # #         cls_embedding = vit_output[:, 0, :]  # Extract CLS token
    # #         embeddings = self.linear_proj(cls_embedding)
            
    # #         out_coords = self.head_coords(embeddings)
    # #         out_clouds = self.head_clouds(embeddings)
    # #         out_buildings = self.head_buildings(embeddings)
    # #         out_landcover = self.head_landcover(embeddings)
            
    # #         return embeddings, vit_output, hidden_states, (out_coords, out_clouds, out_buildings, out_landcover)
    # #         #return embeddings, vit_output, hidden_states
    
    # # # class FoundationViTDecoder(nn.Module):
    # # #     def __init__(
    # # #             self,
    # # #             embed_dim=512,
    # # #             depth=4,
    # # #             num_heads=8,
    # # #             mlp_ratio=4,
    # # #             norm_layer=nn.LayerNorm,
    # # #             chw=(3, 64, 64),
    # # #     ):
    # # #         super().__init__()
            
    # # #         self.chw = chw
    # # #         self.img_size = chw[1]
    # # #         self.embed_dim = embed_dim
    # # #         self.depth = depth
    # # #         self.num_heads = num_heads
    # # #         self.mlp_ratio = mlp_ratio
    # # #         self.norm_layer = norm_layer
            
    # # #         self.decoder_blocks = nn.ModuleList([
    # # #             Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
    # # #             for _ in range(depth)
    # # #         ])
            
    # # #         self.norm = norm_layer(embed_dim)
    # # #         self.reconstruction_layer = nn.Conv2d(
    # # #             in_channels=embed_dim, out_channels=chw[0], kernel_size=1
    # # #         )
        
    # # #     def forward(self, x):
    # # #         for blk in self.decoder_blocks:
    # # #             x = blk(x)
            
    # # #         x = self.norm(x)
    # # #         x = x[:, 1:, :].transpose(1, 2).reshape(-1, self.embed_dim, self.img_size // 4, self.img_size // 4)
    # # #         x = self.reconstruction_layer(x)
    # # #         x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            
    # # #         return x


    # # class FoundationViTDecoder(nn.Module):
    # #     def __init__(
    # #         self,
    # #         chw = (10,128,128), 
    # #         patch_size = 4, 
    # #         output_dim: int = 10, 
    # #         embed_dim: int = 768, 
    # #         decoder_out_channels: int = 256, 
    # #         decoder_in_index: list = [2, 5, 8, 11], 
    # #         decoder_pool_scales: tuple = (1, 2, 3, 6), 
    # #         decoder_norm: dict = {'type': 'BN2d'}
    # #     ):
    # #         super().__init__()

    # #         self.img_size = chw[1]
    # #         self.patch_size = 4
    # #         self.embed_dim = embed_dim
    # #         self.num_classes = output_dim
    # #         self.decoder_out_channels = decoder_out_channels
    # #         self.decoder_in_index = decoder_in_index
    # #         self.decoder_norm = decoder_norm

    # #         self.fpn1 = nn.Sequential(
    # #             nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=self.embed_dim // 2, kernel_size=2, stride=2),
    # #             nn.BatchNorm2d(self.embed_dim // 2),
    # #             nn.ReLU(),
    # #             nn.ConvTranspose2d(in_channels=self.embed_dim // 2, out_channels=self.embed_dim // 4, kernel_size=2, stride=2)
    # #         )

    # #         self.fpn2 = nn.Sequential(
    # #             nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=self.embed_dim // 2, kernel_size=2, stride=2),
    # #             nn.BatchNorm2d(self.embed_dim // 2),
    # #             nn.ReLU()
    # #         )

    # #         self.fpn3 = nn.Identity()
    # #         self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

    # #         self.sample_list_base = nn.ModuleList([self.fpn1, self.fpn2, self.fpn3, self.fpn4])

    # #         self.decoder_upernet = UPerHead(
    # #             in_channels=[self.embed_dim // 4, self.embed_dim // 2, self.embed_dim, self.embed_dim],
    # #             channels=self.decoder_out_channels,
    # #             num_classes=self.num_classes,
    # #             norm_cfg=self.decoder_norm,
    # #             in_index=self.decoder_in_index
    # #         )

    # #     def reshape_vit_features(self, input):
    # #         B, N, D = input.shape
    # #         # B = batch_size, N = number of patches, D = embedding dimension
    # #         # Reshape to obtain spatial resolutions, i.e., (B, N, D) -> (B, H/P, W/P, D)
    # #         H_p = self.img_size // self.patch_size
    # #         W_p = self.img_size // self.patch_size
    # #         input = input.view(B, H_p, W_p, D)
    # #         # Permute to (B, D, H/P, W/P), required for UPerNet
    # #         input = input.permute(0, 3, 1, 2)
    # #         return input

    # #     def forward(self, hidden_states):
    # #         for i, sample in zip(self.decoder_in_index, self.sample_list_base):
    # #             hidden_states[i] = sample(self.reshape_vit_features(hidden_states[i][:, 1:, :]))
    # #         outputs = self.decoder_upernet(hidden_states)
    # #         return outputs

    # # class FoundationViT(nn.Module):     
    # #     def __init__(
    # #             self,
    # #             input_dim=10,
    # #             output_dim=None,
    # #             chw=(10, 128, 128),
    # #             patch_size=4,
    # #             embed_dim=768,
    # #             depth=12,
    # #             num_heads=16,
    # #             mlp_ratio=4,
    # #             norm_layer=nn.LayerNorm,
    # #             latent_dim=512,
    # #             dropout=None,
    # #             activation=nn.LeakyReLU()
    # #     ):
    # #         super().__init__()

    # #         self.input_dim = input_dim
    # #         self.output_dim = input_dim if output_dim is None else output_dim
    # #         self.latent_dim = latent_dim
    # #         self.activation = activation

    # #         self.stem = CNNBlock(
    # #             input_dim,
    # #             chw[0],
    # #             chw=chw,
    # #             activation=self.activation
    # #         )

    # #         self.encoder = FoundationViTEncoder(
    # #             chw=chw,
    # #             patch_size=patch_size,
    # #             embed_dim=embed_dim,
    # #             depth=depth,
    # #             num_heads=num_heads,
    # #             mlp_ratio=mlp_ratio,
    # #             norm_layer=norm_layer,
    # #             latent_dim=latent_dim
    # #         )

    # #         # self.decoder = FoundationViTDecoder( # HARD CODED
    # #         #     embed_dim=embed_dim,
    # #         #     depth=8,
    # #         #     num_heads=4,
    # #         #     mlp_ratio=mlp_ratio,
    # #         #     norm_layer=norm_layer,
    # #         #     chw=chw
    # #         # )
            
    # #         self.decoder = FoundationViTDecoder( # HARD CODED
    # #             chw = chw,
    # #             patch_size=patch_size,
    # #             output_dim=self.input_dim,
    # #             embed_dim=embed_dim
    # #         )

    # #         self.head = CNNBlock(
    # #             channels_in=chw[0],
    # #             channels_out=self.output_dim,
    # #             chw=[self.output_dim, chw[1], chw[2]],
    # #             activation=self.activation,
    # #         activation_out=nn.Sigmoid()
    # #         )

    # #     def forward(self, x):
    # #         #print(f"before stem: {x.shape}") 
    # #         x = self.stem(x)
    # #         #print(f"after stem: {x.shape}")
    # #         embeddings, vit_output, hidden_states, predictions = self.encoder(x)
    # #         #print(f"after encoder: {vit_output.shape}")
    # #         decoded = self.decoder(vit_output)
    # #         #print(f"after decoder: {decoded.shape}")
    # #         reconstruction = self.head(decoded)
    # #         #print(f"reconstruction after head: {reconstruction.shape}")

    # #         #return reconstruction, embeddings, vit_output, decoded, predictions
    # #         return reconstruction


        
    # from mmcv.cnn import ConvModule     
    # from timm.models.vision_transformer import PatchEmbed, Block  
    # from mmseg.models.decode_heads.decode_head import BaseDecodeHead   
    # from mmseg.models.decode_heads.psp_head import PPM 
    # from mmseg.models.utils.wrappers import resize 
    
    # class UPerHead(BaseDecodeHead):
    #     """Unified Perceptual Parsing for Scene Understanding            

    #     This head is the implementation of `UPerNet
    #     <https://arxiv.org/abs/1807.10221>`_.

    #     Args:
    #         pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
    #             Module applied on the last feature. Default: (1, 2, 3, 6).
    #     """

    #     def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
    #         super().__init__(input_transform='multiple_select', **kwargs)
    #         # PSP Module
    #         self.psp_modules = PPM(
    #             pool_scales,
    #             self.in_channels[-1],
    #             self.channels,
    #             conv_cfg=self.conv_cfg,
    #             norm_cfg=self.norm_cfg,
    #             act_cfg=self.act_cfg,
    #             align_corners=self.align_corners)
    #         self.bottleneck = ConvModule(
    #             self.in_channels[-1] + len(pool_scales) * self.channels,
    #             self.channels,
    #             3,
    #             padding=1,
    #             conv_cfg=self.conv_cfg,
    #             norm_cfg=self.norm_cfg,
    #             act_cfg=self.act_cfg)
    #         # FPN Module
    #         self.lateral_convs = nn.ModuleList()
    #         self.fpn_convs = nn.ModuleList()
    #         for in_channels in self.in_channels[:-1]:  # skip the top layer
    #             l_conv = ConvModule(
    #                 in_channels,
    #                 self.channels,
    #                 1,
    #                 conv_cfg=self.conv_cfg,
    #                 norm_cfg=self.norm_cfg,
    #                 act_cfg=self.act_cfg,
    #                 inplace=False)
    #             fpn_conv = ConvModule(
    #                 self.channels,
    #                 self.channels,
    #                 3,
    #                 padding=1,
    #                 conv_cfg=self.conv_cfg,
    #                 norm_cfg=self.norm_cfg,
    #                 act_cfg=self.act_cfg,
    #                 inplace=False)
    #             self.lateral_convs.append(l_conv)
    #             self.fpn_convs.append(fpn_conv)

    #         self.fpn_bottleneck = ConvModule(
    #             len(self.in_channels) * self.channels,
    #             self.channels,
    #             3,
    #             padding=1,
    #             conv_cfg=self.conv_cfg,
    #             norm_cfg=self.norm_cfg,
    #             act_cfg=self.act_cfg)

    #     def psp_forward(self, inputs):
    #         """Forward function of PSP module."""
    #         x = inputs[-1]
    #         psp_outs = [x]
    #         psp_outs.extend(self.psp_modules(x))
    #         psp_outs = torch.cat(psp_outs, dim=1)
    #         output = self.bottleneck(psp_outs)

    #         return output

    #     def _forward_feature(self, inputs):
    #         """Forward function for feature maps before classifying each pixel with
    #         ``self.cls_seg`` fc.

    #         Args:
    #             inputs (list[Tensor]): List of multi-level img features.

    #         Returns:
    #             feats (Tensor): A tensor of shape (batch_size, self.channels,
    #                 H, W) which is feature map for last layer of decoder head.
    #         """
    #         inputs = self._transform_inputs(inputs)

    #         # build laterals
    #         laterals = [
    #             lateral_conv(inputs[i])
    #             for i, lateral_conv in enumerate(self.lateral_convs)
    #         ]

    #         laterals.append(self.psp_forward(inputs))

    #         #print(laterals[i - 1].shape[2:])
    #         #sadfasdf

    #         # build top-down path
    #         used_backbone_levels = len(laterals)
    #         for i in range(used_backbone_levels - 1, 0, -1):
    #             prev_shape = laterals[i - 1].shape[2:]
    #             laterals[i - 1] = laterals[i - 1] + resize(
    #                 laterals[i],
    #                 size=prev_shape,
    #                 mode='bilinear',
    #                 align_corners=self.align_corners)

    #         # build outputs
    #         fpn_outs = [
    #             self.fpn_convs[i](laterals[i])
    #             for i in range(used_backbone_levels - 1)
    #         ]
    #         # append psp feature
    #         fpn_outs.append(laterals[-1])

    #         for i in range(used_backbone_levels - 1, 0, -1):
    #             fpn_outs[i] = resize(
    #                 fpn_outs[i],
    #                 size=fpn_outs[0].shape[2:],
    #                 mode='bilinear',
    #                 align_corners=self.align_corners)
    #         fpn_outs = torch.cat(fpn_outs, dim=1)
    #         feats = self.fpn_bottleneck(fpn_outs)
    #         return feats

    #     def forward(self, inputs):
    #         """Forward function."""
    #         output = self._forward_feature(inputs)
    #         output = self.cls_seg(output)
    #         return output
    
    # class ViTEncoder(nn.Module):
    #     """ 
    #         VisionTransformer backbone    
    #     """
    #     def __init__(self, chw:tuple=(10, 128, 128), patch_size:int=4, output_dim:int=10,
    #                 embed_dim=768, depth=12, num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm, 
    #                 ):
    #         super().__init__()

    #         # Attributes
    #         self.chw = chw  # (C, H, W)
    #         self.in_c = chw[0]
    #         self.img_size = chw[1]
    #         self.patch_size = patch_size
    #         self.output_dim = output_dim
    #         self.embed_dim = embed_dim
    #         self.depth = depth
    #         self.num_heads = num_heads
    #         self.mlp_ratio = mlp_ratio
    #         self.norm_layer = norm_layer
            
    #         # --------------------------------------------------------------------------
    #         # MAE encoder specifics     
    #         self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.in_c, self.embed_dim)
    #         num_patches = self.patch_embed.num_patches

    #         self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
    #         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
    #                                     requires_grad=False)  # # learnable with sin-cos embedding init      

    #         self.blocks = nn.ModuleList([
    #             Block(self.embed_dim, self.num_heads, self.mlp_ratio, qkv_bias=True, norm_layer= self.norm_layer)
    #             for i in range(self.depth)])
    #         self.norm = self.norm_layer(self.embed_dim)

    #         self.initialize_weights()
    #         # --------------------------------------------------------------------------

    #     def initialize_weights(self):
    #         # initialization
    #         # initialize (and freeze) pos_embed by sin-cos embedding
    #         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
    #                                             cls_token=True)
    #         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    #         # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
    #         w = self.patch_embed.proj.weight.data
    #         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    #         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
    #         torch.nn.init.normal_(self.cls_token, std=.02)

    #         # initialize nn.Linear and nn.LayerNorm
    #         self.apply(self._init_weights)

    #     def _init_weights(self, m):
    #         if isinstance(m, nn.Linear):
    #             # we use xavier_uniform following official JAX ViT:
    #             torch.nn.init.xavier_uniform_(m.weight)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)

    #     def forward(self, x):
    #         # embed patches
    #         # B, C, H, W = x.shape
    #         x = self.patch_embed(x)

    #         # add pos embed w/o cls token
    #         x = x + self.pos_embed[:, 1:, :]

    #         # append cls token
    #         cls_token = self.cls_token + self.pos_embed[:, :1, :]
    #         cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    #         x = torch.cat((cls_tokens, x), dim=1)

    #         # apply Transformer blocks
    #         hidden_states = []
    #         for blk in self.blocks:
    #             x = blk(x)
    #             hidden_states.append(x)
    #         x = self.norm(x)
    #         hidden_states[-1] = x
    #         # # remove cls token
    #         #x = x[:, 1:, :]

    #         return x, hidden_states

    # class ViTUperNet(nn.Module):
    #     """ 
    #     ViT backbone with UperNet decoder head   
    #     """
    #     def __init__(self,  chw:tuple=(10, 128, 128), patch_size:int=4, output_dim:int=11,
    #                 embed_dim=768, depth=12, num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm, 
    #                 decoder_out_channels = 256, decoder_in_index = [2,5,8,11], decoder_pool_scales = (1,2,3,6), decoder_norm= {'type': 'BN2d'}):
    #         super().__init__()

    #         # Attributes
    #         self.chw = chw  # (C, H, W)
    #         self.in_c = chw[0]
    #         self.img_size = chw[1]
    #         self.patch_size = patch_size
    #         self.num_classes = output_dim
    #         self.embed_dim = embed_dim
    #         self.depth = depth
    #         self.num_heads = num_heads
    #         self.mlp_ratio = mlp_ratio
    #         self.norm_layer = norm_layer
    #         self.decoder_in_index = decoder_in_index
    #         self.decoder_out_channels = decoder_out_channels
    #         self.decoder_pool_scales = decoder_pool_scales
    #         self.decoder_norm = decoder_norm
    #         assert all(element < self.depth for element in self.decoder_in_index), f"Please select intermediate features from one of the {self.depth} layers"
        

    #         # --------------------------------------------------------------------------
    #         # # encoder specifics      
    #         self.vit_encoder = ViTEncoder(chw=self.chw, 
    #                                     patch_size=self.patch_size, output_dim=self.num_classes,
    #                                     embed_dim=self.embed_dim, depth=self.depth, num_heads=self.num_heads,
    #                                     mlp_ratio=self.mlp_ratio, norm_layer=self.norm_layer)
    
    #         # --------------------------------------------------------------------------

    #         # --------------------------------------------------------------------------
    #         # decoder UperNet   
    #         # upsample/downsample the input before feeding it to UperNet
    #         self.fpn1 = nn.Sequential(nn.ConvTranspose2d(in_channels = self.embed_dim, out_channels = self.embed_dim//2, kernel_size= 2, stride= 2),
    #                                 nn.BatchNorm2d(self.embed_dim//2), nn.ReLU(),
    #                                 nn.ConvTranspose2d(in_channels=self.embed_dim//2, out_channels= self.embed_dim//4, kernel_size= 2, stride= 2))
    #         self.fpn2 = nn.Sequential(nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels= self.embed_dim//2, kernel_size= 2, stride = 2),
    #                                 nn.BatchNorm2d(self.embed_dim//2), nn.ReLU())
             
    #         self.fpn3 = nn.Identity()  
    #         self.fpn4 = nn.MaxPool2d(kernel_size= 2, stride = 2) 
            
    #         self.sample_list_base = nn.ModuleList([self.fpn1, self.fpn2, self.fpn3, self.fpn4])
    #         self.decoder_upernet = UPerHead(in_channels =[self.embed_dim//4, self.embed_dim//2, self.embed_dim, self.embed_dim] , channels = self.decoder_out_channels, 
    #                                 num_classes = self.num_classes, norm_cfg = self.decoder_norm, in_index = self.decoder_in_index)
 
    #         # self.learnable_upsample = nn.Sequential(
    #         #     nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=4, stride=4, padding=0),
    #         #     nn.BatchNorm2d(self.num_classes),
    #         #     nn.ReLU(inplace=True)
    #         # )  
 
    #         # --------------------------------------------------------------------------  
            
        
    #     def reshape_vit_features(self, input):
    #         B,N,D = input.shape
    #         # B = batch_size , N = number of patches, D = embedding dimension  
    #         # Reshape to obtain spatial resolutions, i.e. (B, N, D) -> (B, H/P, W/P, D)
    #         H_p = self.img_size // self.patch_size
    #         W_p = self.img_size// self.patch_size
    #         input = input.view(B, H_p, W_p, D)
    #         # Permute to (B, D, H/P, W/P), i.e. needed for UPerNet
    #         input = input.permute(0, 3, 1, 2)
    #         return input
            

    #     def forward(self, x):
    #         # B, N, D = hidden_states[i].shape       
    #         _, hidden_states = self.vit_encoder(x)
            
    #         #print(hidden_states.shape)  
    #         #sadfasdf
            
    #         # select desired intermediate features: remove cls token + reshape to appropriate size + upsample/downsample + extract their dimensions   
    #         for i, sample in zip(self.decoder_in_index, self.sample_list_base):
    #             hidden_states[i] = sample(self.reshape_vit_features(hidden_states[i][:,1:, :]))
    #         # decoder
    #         outputs = self.decoder_upernet(hidden_states)
            
            
            
            
            
            
            
    #         #outputs = self.learnable_upsample(outputs)  
            
            
            
            
            
            
            
            
    #         return outputs
        
        
    # def vit_upernet_large(**kwargs):
    #     model = ViTUperNet(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, decoder_in_index=[5,11,17,23],
    #                 norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    #     return model

    # def vit_upernet_pretrained(checkpoint, chw:tuple=(10, 128, 128), patch_size:int=4, output_dim:int=11, freeze_body = True, **kwargs):
        
        
    #     # # load pre-trained model weights       
    #     model = vit_upernet_large(chw = chw, patch_size = patch_size, output_dim = output_dim, **kwargs) 
    #     msg = model.vit_encoder.load_state_dict(checkpoint, strict= False)
    #     print(msg)
        
    #     if freeze_body:
    #         for _, param in model.vit_encoder.named_parameters():
    #                 param.requires_grad = False

    #     return model 
    
    # # model = ViTUperNet(chw:tuple=(10, 128, 128), patch_size:int=4, output_dim:int=11,
    # #                 embed_dim=768, depth=12, num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm, 
    # #                 decoder_out_channels = 256, decoder_in_index = [2,5,8,11], decoder_pool_scales = (1,2,3,6), decoder_norm= {'type': 'BN2d'}))       
    
    # #model = ViTUperNet()       
    
    # #model = FoundationViT(
    # model = ViTUperNet(
    #         #input_dim=10,
    #         chw=(10, 128, 128),
    #         patch_size=4,
    #         embed_dim=512,
    #         depth=32,
    #         num_heads=16,
    #         mlp_ratio=4,
    #         norm_layer=nn.LayerNorm,
    #         #latent_dim=1024,
    #         #dropout=None,
    #         #activation=nn.LeakyReLU()
    #         decoder_in_index = [28,29,30,31],
    #         output_dim = 1
    #     )



    # from mmcv.cnn import ConvModule           
    # from mmseg.models.utils.wrappers import resize
    # from mmseg.models.decode_heads.decode_head import BaseDecodeHead
    # from mmseg.models.decode_heads.psp_head import PPM
    
    # from timm.models.vision_transformer import PatchEmbed, Block
    
    
    # class ScaleSkip2D(nn.Module):
    #     """
    #     Learnable channel-wise scale and bias for skip connections        

    #     Parameters
    #     ----------
    #     channels : int
    #         Number of channels in the input

    #     drop_y : float
    #         Probability of dropping a channel in the skip connection.
    #         Drops are replaces with Gaussian noise.

    #     signal_to_noise : tuple or None
    #         Range of signal to noise ratios to use for the dropped channels. 0.0 is pure noise, 1.0 is pure signal.
    #         The amount of signal is randomly sampled from this range for each channel.
    #         If None, no signal is added to the dropped channels.
    #         default: (0.1, 0.9)

    #     size : float
    #         Standard deviation of the normal distribution to sample initial values from. 
    #         default: 0.01
    #     """

    #     def __init__(self, channels, drop_y=None, signal_to_noise=(0.1, 0.9), size=0.01):
    #         super(ScaleSkip2D, self).__init__()
    #         self.channels = channels
    #         self.drop_y = drop_y
    #         self.size = size

    #         # Learnable scale and bias  
    #         self.x_skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1))
    #         self.y_skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1))
    #         self.y_skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
    #         self.x_skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))

    #         if self.drop_y is not None and self.drop_y > 0.0:
    #             self.drop_y = GaussianDropout2d(self.drop_y, signal_to_noise=signal_to_noise)
    #         else:
    #             self.drop_y = None

    #         self.set_weights()
    #         while torch.any(self.x_skipscale == 0) or torch.any(self.y_skipscale == 0) or torch.any(
    #                 self.y_skipbias == 0
    #         ) or torch.any(self.x_skipbias == 0):
    #             self.set_weights()

    #     def set_weights(self):
    #         nn.init.trunc_normal_(self.x_skipscale, 1.0, self.size)
    #         nn.init.trunc_normal_(self.y_skipscale, 1.0, self.size)
    #         nn.init.trunc_normal_(self.y_skipbias, 0.0, self.size)
    #         nn.init.trunc_normal_(self.x_skipbias, 0.0, self.size)

    #     def forward(self, x, y):
    #         x = (x * self.x_skipscale) + self.x_skipbias
    #         y = (y * self.y_skipscale) + self.y_skipbias

    #         if self.drop_y is not None:
    #             y = self.drop_y(y)

    #         return x + y
    
    
    # class SE_Block(nn.Module):
    #     """ credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4 """

    #     def __init__(self, channels, reduction=16):
    #         super().__init__()
    #         self.reduction = reduction
    #         self.squeeze = nn.AdaptiveAvgPool2d(1)
    #         self.excitation = nn.Sequential(
    #             nn.Linear(channels, max(1, channels // self.reduction), bias=False),
    #             nn.GELU(),
    #             nn.Linear(max(1, channels // self.reduction), channels, bias=False),
    #             nn.Sigmoid(),
    #         )

    #     def forward(self, x):
    #         bs, c, _, _ = x.shape
    #         y = self.squeeze(x).view(bs, c)
    #         y = self.excitation(y).view(bs, c, 1, 1)

    #         return x * y.expand_as(x)
    
    
    # class CNNBlock(nn.Module):
    #     """
    #     This is a standard CNN block with a 1x1 convolutional matcher for the skip connection. 
    #     It adds a learnable scale and bias to the skip connection.

    #     Parameters
    #     ----------
    #     channels_in : int
    #         Number of channels in the input

    #     channels_out : int or None
    #         Number of channels in the output. If None, the number of channels is unchanged.
    #         default: None

    #     group_size : int
    #         Number of groups for the 3x3 convolution.
    #         default: 1

    #     activation : torch.nn.Module
    #         Activation function to use after the first convolution.
    #         default: torch.nn.GELU()

    #     activation_out : torch.nn.Module or None
    #         Activation function to use after the last convolution.
    #         If None, the same activation as the first convolution is used.
    #         default: None

    #     chw : tuple or None
    #         Height and width of the input. If None, batch norm is used instead of layer norm.
    #         default: None
    #     """

    #     def __init__(
    #             self,
    #             channels_in,
    #             channels_out=None,
    #             chw=None,
    #             group_size=1,
    #             activation=nn.GELU(),
    #             activation_out=None,
    #             residual=True,
    #             reduction=1,
    #     ):
    #         super().__init__()

    #         assert chw is not None, "chw must be specified"

    #         self.channels_in = channels_in
    #         self.channels_out = channels_in if channels_out is None else channels_out
    #         self.channels_internal = self.channels_out // reduction
    #         self.chw = chw
    #         self.group_size = group_size
    #         self.activation = activation
    #         self.activation_out = activation if activation_out is None else activation_out
    #         self.residual = residual
    #         self.reduction = reduction
    #         self.squeeze = SE_Block(self.channels_out, 16)

    #         self.matcher = nn.Conv2d(
    #             self.channels_in, self.channels_out, 1, padding=0,
    #             bias=False
    #             ) if self.channels_in != self.channels_out else None

    #         self.norm1 = nn.LayerNorm([self.channels_internal, self.chw[1], self.chw[2]])
    #         self.norm2 = nn.LayerNorm([self.channels_internal, self.chw[1], self.chw[2]])

    #         self.conv1 = nn.Conv2d(self.channels_in, self.channels_internal, 1, padding=0, bias=False)
    #         self.conv2 = nn.Conv2d(
    #             self.channels_internal, self.channels_internal, 3, padding=1, groups=self.group_size,
    #             bias=False, padding_mode="replicate"
    #             )
    #         self.conv3 = nn.Conv2d(self.channels_internal, self.channels_out, 1, padding=0, bias=True)

    #         self.scaler = ScaleSkip2D(self.channels_out) if self.residual else None
    #         #self.scaler2 = ScaleSkip2D(self.channels_out) if self.residual else None

    #     def forward(self, x):
    #         identity = x if self.matcher is None else self.matcher(x)

    #         x = self.conv1(x) 
    #         x = self.norm1(x)
    #         x = self.activation(x)

    #         x = self.conv2(x)
    #         x = self.norm2(x)
    #         x = self.activation(x)

    #         x = self.conv3(x)
    #         x = self.squeeze(x)

    #         if self.residual:
    #             x = self.scaler(x, identity)
    #             #x = self.scaler2(x, identity)

    #         x = self.activation_out(x)

    #         #return x
    #         #return x.long() 
    #         return x

    
    # class UPerHead(BaseDecodeHead):
    #     """Unified Perceptual Parsing for Scene Understanding                     

    #     This head is the implementation of `UPerNet
    #     <https://arxiv.org/abs/1807.10221>`_.

    #     Args:
    #         pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
    #             Module applied on the last feature. Default: (1, 2, 3, 6).
    #     """

    #     def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
    #         super().__init__(input_transform='multiple_select', **kwargs)
    #         # PSP Module 
    #         self.psp_modules = PPM(
    #             pool_scales,
    #             self.in_channels[-1],
    #             self.channels,
    #             conv_cfg=self.conv_cfg,
    #             norm_cfg=self.norm_cfg,
    #             act_cfg=self.act_cfg,
    #             align_corners=self.align_corners)
    #         self.bottleneck = ConvModule(
    #             self.in_channels[-1] + len(pool_scales) * self.channels,
    #             self.channels,
    #             3,
    #             padding=1,
    #             conv_cfg=self.conv_cfg,
    #             norm_cfg=self.norm_cfg,
    #             act_cfg=self.act_cfg)
    #         # FPN Module
    #         self.lateral_convs = nn.ModuleList()
    #         self.fpn_convs = nn.ModuleList()
    #         for in_channels in self.in_channels[:-1]:  # skip the top layer
    #             l_conv = ConvModule(
    #                 in_channels,
    #                 self.channels,
    #                 1,
    #                 conv_cfg=self.conv_cfg,
    #                 norm_cfg=self.norm_cfg,
    #                 act_cfg=self.act_cfg,
    #                 inplace=False)
    #             fpn_conv = ConvModule(
    #                 self.channels,
    #                 self.channels,
    #                 3,
    #                 padding=1,
    #                 conv_cfg=self.conv_cfg,
    #                 norm_cfg=self.norm_cfg,
    #                 act_cfg=self.act_cfg,
    #                 inplace=False)
    #             self.lateral_convs.append(l_conv)
    #             self.fpn_convs.append(fpn_conv)

    #         self.fpn_bottleneck = ConvModule(
    #             len(self.in_channels) * self.channels,
    #             self.channels,
    #             3,
    #             padding=1,
    #             conv_cfg=self.conv_cfg,
    #             norm_cfg=self.norm_cfg,
    #             act_cfg=self.act_cfg)

    #     def psp_forward(self, inputs):
    #         """Forward function of PSP module."""
    #         x = inputs[-1]
    #         psp_outs = [x]
    #         psp_outs.extend(self.psp_modules(x))
    #         psp_outs = torch.cat(psp_outs, dim=1)
    #         output = self.bottleneck(psp_outs)

    #         return output

    #     def _forward_feature(self, inputs):
    #         """Forward function for feature maps before classifying each pixel with
    #         ``self.cls_seg`` fc.

    #         Args:
    #             inputs (list[Tensor]): List of multi-level img features.

    #         Returns:
    #             feats (Tensor): A tensor of shape (batch_size, self.channels,
    #                 H, W) which is feature map for last layer of decoder head.
    #         """
    #         inputs = self._transform_inputs(inputs)

    #         # build laterals
    #         laterals = [
    #             lateral_conv(inputs[i])
    #             for i, lateral_conv in enumerate(self.lateral_convs)
    #         ]

    #         laterals.append(self.psp_forward(inputs))

    #         # build top-down path
    #         used_backbone_levels = len(laterals)
    #         for i in range(used_backbone_levels - 1, 0, -1):
    #             prev_shape = laterals[i - 1].shape[2:]
    #             laterals[i - 1] = laterals[i - 1] + resize(
    #                 laterals[i],
    #                 size=prev_shape,
    #                 mode='bilinear',
    #                 align_corners=self.align_corners)

    #         # build outputs
    #         fpn_outs = [
    #             self.fpn_convs[i](laterals[i])
    #             for i in range(used_backbone_levels - 1)
    #         ]
    #         # append psp feature
    #         fpn_outs.append(laterals[-1])

    #         for i in range(used_backbone_levels - 1, 0, -1):
    #             fpn_outs[i] = resize(
    #                 fpn_outs[i],
    #                 size=fpn_outs[0].shape[2:],
    #                 mode='bilinear',
    #                 align_corners=self.align_corners)
    #         fpn_outs = torch.cat(fpn_outs, dim=1)
    #         feats = self.fpn_bottleneck(fpn_outs)
    #         return feats

    #     def forward(self, inputs):
    #         """Forward function."""
    #         output = self._forward_feature(inputs)
    #         output = self.cls_seg(output)
    #         return output
    
    
    # def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    #     """
    #     grid_size: int of the grid height and width
    #     return:
    #     pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    #     """
    #     grid_h = np.arange(grid_size, dtype=np.float32)
    #     grid_w = np.arange(grid_size, dtype=np.float32)
    #     grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    #     grid = np.stack(grid, axis=0)

    #     grid = grid.reshape([2, 1, grid_size, grid_size])
    #     pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    #     if cls_token:
    #         pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    #     return pos_embed


    # def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    #     assert embed_dim % 2 == 0

    #     # use half of dimensions to encode grid_h
    #     emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    #     emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    #     emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    #     return emb


    # def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    #     """
    #     embed_dim: output dimension for each position
    #     pos: a list of positions to be encoded: size (M,)
    #     out: (M, D)
    #     """
    #     assert embed_dim % 2 == 0
    #     omega = np.arange(embed_dim // 2, dtype=float)
    #     omega /= embed_dim / 2.
    #     omega = 1. / 10000**omega  # (D/2,)

    #     pos = pos.reshape(-1)  # (M,)
    #     out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    #     emb_sin = np.sin(out) # (M, D/2)
    #     emb_cos = np.cos(out) # (M, D/2)

    #     emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    #     return emb


    # def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    #     """
    #     embed_dim: output dimension for each position
    #     pos: a list of positions to be encoded: size (M,)
    #     out: (M, D)
    #     """
    #     assert embed_dim % 2 == 0
    #     omega = torch.arange(embed_dim // 2, dtype=np.float, device=device)
    #     omega /= embed_dim / 2.
    #     omega = 1. / 10000**omega  # (D/2,)

    #     pos = pos.reshape(-1)  # (M,)
    #     out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    #     emb_sin = torch.sin(out) # (M, D/2)
    #     emb_cos = torch.cos(out) # (M, D/2)

    #     emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    #     return emb.double()

    # def interpolate_pos_embed(model, checkpoint_model): 
    #     if 'pos_embed' in checkpoint_model:
    #         pos_embed_checkpoint = checkpoint_model['pos_embed']
    #         embedding_size = pos_embed_checkpoint.shape[-1]
    #         try:
    #             num_patches = model.patch_embed.num_patches
    #         except AttributeError as err:
    #             num_patches = model.patch_embed[0].num_patches
    #         num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    #         # height (== width) for the checkpoint position embedding
    #         orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    #         # height (== width) for the new position embedding
    #         new_size = int(num_patches ** 0.5)
    #         # class_token and dist_token are kept unchanged
    #         if orig_size != new_size:
    #             print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
    #             extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    #             # only the position tokens are interpolated
    #             pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    #             pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    #             pos_tokens = torch.nn.functional.interpolate(
    #                 pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    #             pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    #             new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    #             checkpoint_model['pos_embed'] = new_pos_embed

    # class ViTEncoder(nn.Module):
    #     """
    #         VisionTransformer backbone   
    #     """
    #     def __init__(self, chw: tuple = (10, 128, 128), patch_size: int = 4, output_dim: int = 10,
    #                 embed_dim=768, depth=12, num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm,
    #                 ):
    #         super().__init__()

    #         # Attributes
    #         self.chw = chw  # (C, H, W)
    #         self.in_c = chw[0]
    #         self.img_size = chw[1]
    #         self.patch_size = patch_size
    #         self.output_dim = output_dim
    #         self.embed_dim = embed_dim
    #         self.depth = depth
    #         self.num_heads = num_heads
    #         self.mlp_ratio = mlp_ratio
    #         self.norm_layer = norm_layer

    #         # --------------------------------------------------------------------------
    #         self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.in_c, self.embed_dim)
    #         num_patches = self.patch_embed.num_patches

    #         self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
    #         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
    #                                     requires_grad=False)  # # learnable with sin-cos embedding init  

    #         self.blocks = nn.ModuleList([
    #             Block(self.embed_dim, self.num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer)
    #             for i in range(self.depth)])
    #         self.norm = self.norm_layer(self.embed_dim)

    #         self.initialize_weights()
    #         # --------------------------------------------------------------------------

    #     def initialize_weights(self):
    #         # initialization
    #         # initialize (and freeze) pos_embed by sin-cos embedding
    #         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
    #                                             cls_token=True)
    #         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    #         # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
    #         w = self.patch_embed.proj.weight.data
    #         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    #         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
    #         torch.nn.init.normal_(self.cls_token, std=.02)

    #         # initialize nn.Linear and nn.LayerNorm
    #         self.apply(self._init_weights)

    #     def _init_weights(self, m):
    #         if isinstance(m, nn.Linear):
    #             # we use xavier_uniform following official JAX ViT:
    #             torch.nn.init.xavier_uniform_(m.weight)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)

    #     def forward(self, x):
    #         # embed patches
    #         # B, C, H, W = x.shape
    #         x = self.patch_embed(x)

    #         # add pos embed w/o cls token
    #         x = x + self.pos_embed[:, 1:, :]

    #         # append cls token
    #         cls_token = self.cls_token + self.pos_embed[:, :1, :]
    #         cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    #         x = torch.cat((cls_tokens, x), dim=1)

    #         # apply Transformer blocks
    #         hidden_states = []
    #         for blk in self.blocks:
    #             x = blk(x)
    #             hidden_states.append(x)
    #         x = self.norm(x)
    #         hidden_states[-1] = x
    #         # # remove cls token
    #         # x = x[:, 1:, :]

    #         return x, hidden_states  

    # class FoundationViTEncoder(nn.Module):   
    #     def __init__(
    #             self,
    #             chw=(3, 64, 64),  # Default image size
    #             patch_size=4,
    #             embed_dim=768,
    #             depth=12,
    #             num_heads=16,
    #             mlp_ratio=4,
    #             norm_layer=nn.LayerNorm,
    #             latent_dim=512
    #     ):
    #         super().__init__() 

    #         self.vit_encoder = ViTEncoder(
    #             chw=chw,
    #             patch_size=patch_size,
    #             embed_dim=embed_dim,
    #             depth=depth,
    #             num_heads=num_heads,
    #             mlp_ratio=mlp_ratio,
    #             norm_layer=norm_layer
    #         )

    #         self.latent_dim = latent_dim
    #         self.linear_proj = nn.Linear(embed_dim, latent_dim)

    #         self.head_clouds = nn.Linear(latent_dim, 4)
    #         self.head_landcover = nn.Linear(latent_dim, 11)
    #         self.head_buildings = nn.Sequential(
    #             nn.Linear(latent_dim, 1),
    #             nn.Sigmoid()
    #         )
    #         self.head_coords = nn.Sequential(
    #             nn.Linear(latent_dim, 4),
    #             nn.Sigmoid()
    #         )

    #     def forward(self, x):
    #         vit_output, hidden_states = self.vit_encoder(x)  # Extract ViT embeddings   
    #         cls_embedding = vit_output[:, 0, :]  # Extract CLS token 
    #         embeddings = self.linear_proj(cls_embedding)

    #         out_coords = self.head_coords(embeddings)
    #         out_clouds = self.head_clouds(embeddings)
    #         out_buildings = self.head_buildings(embeddings)
    #         out_landcover = self.head_landcover(embeddings)

    #         return embeddings, vit_output, hidden_states, (out_coords, out_clouds, out_buildings, out_landcover)

    # class FoundationViTDecoder(nn.Module):   
    #     def __init__(
    #         self,
    #         chw = (10,128,128), 
    #         patch_size = 4, 
    #         output_dim: int = 10, 
    #         embed_dim: int = 768, 
    #         decoder_out_channels: int = 256, 
    #         #decoder_in_index: list = [2, 5, 8, 11],   
    #         #decoder_in_index: list = [2, 5, 6, 7], 
    #         #decoder_in_index: list = [1, 2, 3, 4], 
    #         decoder_in_index: list = [2, 3, 4, 5], 
    #         decoder_pool_scales: tuple = (1, 2, 3, 6), 
    #         decoder_norm: dict = {'type': 'BN2d'}
    #     ):
    #         super().__init__() 

    #         self.img_size = chw[1]
    #         self.patch_size = 4
    #         self.embed_dim = embed_dim
    #         self.num_classes = output_dim
    #         self.decoder_out_channels = decoder_out_channels
    #         self.decoder_in_index = decoder_in_index
    #         self.decoder_norm = decoder_norm

    #         self.fpn1 = nn.Sequential(
    #             nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=self.embed_dim // 2, kernel_size=2, stride=2),
    #             nn.BatchNorm2d(self.embed_dim // 2),
    #             nn.ReLU(),
    #             nn.ConvTranspose2d(in_channels=self.embed_dim // 2, out_channels=self.embed_dim // 4, kernel_size=2, stride=2)
    #         )

    #         self.fpn2 = nn.Sequential(
    #             nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=self.embed_dim // 2, kernel_size=2, stride=2),
    #             nn.BatchNorm2d(self.embed_dim // 2),
    #             nn.ReLU()
    #         )

    #         self.fpn3 = nn.Identity() 
    #         self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

    #         self.sample_list_base = nn.ModuleList([self.fpn1, self.fpn2, self.fpn3, self.fpn4])

    #         self.decoder_upernet = UPerHead(
    #             in_channels=[self.embed_dim // 4, self.embed_dim // 2, self.embed_dim, self.embed_dim],
    #             channels=self.decoder_out_channels,
    #             num_classes=self.num_classes,
    #             norm_cfg=self.decoder_norm,
    #             in_index=self.decoder_in_index
    #         )
            
    #         #self.learnable_upsample = nn.Sequential(
    #         #    nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=4, stride=4, padding=0),
    #         #    nn.BatchNorm2d(self.num_classes),
    #         #    nn.ReLU(inplace=True)
    #         #)    

    #     def reshape_vit_features(self, input): 
    #         B, N, D = input.shape
    #         # B = batch_size, N = number of patches, D = embedding dimension  
    #         # Reshape to obtain spatial resolutions, i.e., (B, N, D) -> (B, H/P, W/P, D)
    #         H_p = self.img_size // self.patch_size
    #         W_p = self.img_size // self.patch_size
    #         input = input.view(B, H_p, W_p, D)
    #         # Permute to (B, D, H/P, W/P), required for UPerNet 
    #         input = input.permute(0, 3, 1, 2)
    #         return input

    #     def forward(self, hidden_states):
    #         for i, sample in zip(self.decoder_in_index, self.sample_list_base):
    #             hidden_states[i] = sample(self.reshape_vit_features(hidden_states[i][:, 1:, :]))
    #         outputs = self.decoder_upernet(hidden_states)
            
    #         #outputs = self.learnable_upsample(outputs)    
            
    #         return outputs

    # class PhilEO_ViT(nn.Module):  
    #     def __init__(
    #             self,
    #             input_dim=10,
    #             output_dim=10,
    #             chw=(10, 128, 128),
    #             patch_size=4,
    #             embed_dim=768,
    #             depth=12,
    #             num_heads=16,
    #             mlp_ratio=4,
    #             norm_layer=nn.LayerNorm,
    #             latent_dim=512,
    #             dropout=None,
    #             activation=nn.LeakyReLU()
    #     ):
    #         super().__init__() 

    #         self.input_dim = input_dim
    #         self.output_dim = input_dim if output_dim is None else output_dim
    #         self.latent_dim = latent_dim
    #         self.embed_dim = embed_dim
    #         self.activation = activation

    #         self.stem = CNNBlock(
    #             self.input_dim,
    #             chw[0],
    #             chw=chw,
    #             activation=self.activation
    #         )

    #         self.encoder = FoundationViTEncoder(
    #             chw=chw,
    #             patch_size=patch_size,
    #             embed_dim=self.embed_dim,
    #             depth=depth,
    #             num_heads=num_heads,
    #             mlp_ratio=mlp_ratio,
    #             norm_layer=norm_layer,
    #             latent_dim=self.latent_dim
    #         )

    #         self.decoder = FoundationViTDecoder( # # HARD CODED   
    #             chw = chw,
    #             patch_size=patch_size,
    #             output_dim=self.input_dim,
    #             embed_dim=self.embed_dim
    #         )
            
    #         # self.head = CNNBlock(
    #         #     channels_in=chw[0],
    #         #     channels_out=self.output_dim,
    #         #     chw=[self.output_dim, chw[1], chw[2]],
    #         #     activation=self.activation,
    #         #     activation_out=nn.Sigmoid()
    #         # )
    #         self.head2 = CNNBlock(
    #             channels_in=chw[0],
    #             channels_out=self.output_dim,
    #             chw=[self.output_dim, chw[1], chw[2]],
    #             activation=self.activation,
    #             activation_out=nn.Sigmoid()
    #         )

    #     def forward(self, x):
    #         x = self.stem(x)
    #         embeddings, vit_output, hidden_states, predictions = self.encoder(x)
    #         decoded = self.decoder(hidden_states)
            
    #         #reconstruction = self.head(decoded)
    #         reconstruction = self.head2(decoded)

    #         #return reconstruction, embeddings, vit_output, decoded, predictions 
    #         return reconstruction

    # #model = PhilEO_ViT()                                       
    # model = PhilEO_ViT(depth=6, output_dim=11) 
    
    # model.train()   



    # #import sys                                                     
    # #sys.path.append('./2DMamba/2DMambaMIL')     
    
    # from models.MambaMIL_2D import MambaMIL_2D 
    # #from 2DMamba.2DMambaMIL.models.MambaMIL_2D import MambaMIL_2D
    
    # from argparse import Namespace     
    # args = Namespace(bag_loss='nll_surv', cuda_pscan=False, data_root_dir=None, debug=False, device=0, drop_out=0.25, early_stopping=False, 
    #                  exp_code='demo', fold=0, h5_path=None, in_dim=128, input_type='feature_uni', k=1, k_end=-1, k_start=-1, label_frac=1.0, 
    #                  log_data=False, lr=0.0001, mamba_2d=False, mamba_2d_pad_token='trainable', mamba_2d_patch_size=4, mambamil_dim=512, 
    #                  mambamil_inner_layernorms=False, mambamil_layer=1, mambamil_rate=10, mambamil_state_dim=16, mambamil_type=None, max_epochs=20, 
    #                  mode='path', model_type='MambaMIL_2D', opt='adamw', patch_encoder='feature_uni', patch_encoder_batch_size=128, patch_size='', 
    #                  pos_emb_dropout=0.0, pos_emb_type=None, preloading='no', pscan=True, reg=1e-05, results_dir='results/demo_s1', reverse_ord=False, 
    #                  seed=1, shuffle_patch=False, split_dir='./splits/BRACS_100', survival=False, task='BRACS', testing=False, train_patch_encoder=False, weighted_sample=False)
    
    # model = MambaMIL_2D(args=args) 


    
    # from models.MambaMIL_2D import MambaMIL_2D 
    # #from 2DMamba.2DMambaMIL.models.MambaMIL_2D import MambaMIL_2D
    
    # from argparse import Namespace     
    # args = Namespace(bag_loss='nll_surv', cuda_pscan=False, data_root_dir=None, debug=False, device=0, drop_out=0.25, early_stopping=False, 
    #                  exp_code='demo', fold=0, h5_path=None, in_dim=128, input_type='feature_uni', k=1, k_end=-1, k_start=-1, label_frac=1.0, 
    #                  log_data=False, lr=0.0001, mamba_2d=False, mamba_2d_pad_token='trainable', mamba_2d_patch_size=4, mambamil_dim=512, 
    #                  mambamil_inner_layernorms=False, mambamil_layer=4, mambamil_rate=10, mambamil_state_dim=16, mambamil_type=None, max_epochs=20, 
    #                  mode='path', model_type='MambaMIL_2D', opt='adamw', patch_encoder='feature_uni', patch_encoder_batch_size=128, patch_size='', 
    #                  pos_emb_dropout=0.0, pos_emb_type=None, preloading='no', pscan=True, reg=1e-05, results_dir='results/demo_s1', reverse_ord=False, 
    #                  seed=1, shuffle_patch=False, split_dir='./splits/BRACS_100', survival=False, task='BRACS', testing=False, train_patch_encoder=False, weighted_sample=False)
    
    # model = MambaMIL_2D(args=args)             
    
    # # # FLOPs: 18.2 G 
    # # # 9626252


    
    import terratorch # # even though we don't use the import directly, we need it so that the models are available in the timm registry            
    from terratorch.models import EncoderDecoderFactory  
    from terratorch.datasets import HLSBands

    model_factory = EncoderDecoderFactory()  
    #model_factory = TERRATORCH_BACKBONE_REGISTRY     

    from terratorch.registry import BACKBONE_REGISTRY, TERRATORCH_BACKBONE_REGISTRY, TERRATORCH_DECODER_REGISTRY

    # print([backbone    
    # for backbone in TERRATORCH_BACKBONE_REGISTRY 
    # if 'terramind_v1' in backbone])

    # # # ['terramind_v1_base', 'terramind_v1_base_tim', 'terramind_v1_large', 'terramind_v1_large_tim']      

    # model = BACKBONE_REGISTRY.build(
    #     "terramind_v1_base",
    #     #modalities=["S2L1C", "S1GRD"],
    #     modalities=["S2L2A"],
    #     pretrained=True,
    # ) 

    # # Segmentation mask that build the model and handles training and validation steps.  
    # model = terratorch.tasks.SemanticSegmentationTask(
    #     model_factory="EncoderDecoderFactory",  # Combines a backbone with necks, the decoder, and a head
    #     model_args={
    #         # TerraMind backbone
    #         "backbone": "terramind_v1_base", # large version: terramind_v1_large 
    #         "backbone_pretrained": True,
    #         "backbone_modalities": ["S2L2A"],
    #         # Optionally, define the input bands. This is only needed if you select a subset of the pre-training bands, as explained above.
    #         # "backbone_bands": {"S1GRD": ["VV"]},
            
    #         # Necks 
    #         "necks": [
    #             {
    #                 "name": "SelectIndices",
    #                 "indices": [2, 5, 8, 11] # indices for terramind_v1_base
    #                 # "indices": [5, 11, 17, 23] # indices for terramind_v1_large
    #             },
    #             {"name": "ReshapeTokensToImage",
    #             "remove_cls_token": False},  # TerraMind is trained without CLS token, which neads to be specified.
    #             {"name": "LearnedInterpolateToPyramidal"}  # Some decoders like UNet or UperNet expect hierarchical features. Therefore, we need to learn a upsampling for the intermediate embedding layers when using a ViT like TerraMind.
    #         ],
            
    #         # Decoder
    #         "decoder": "UNetDecoder",
    #         "decoder_channels": [512, 256, 128, 64],
            
    #         # Head
    #         "head_dropout": 0.1,
    #         "num_classes": 2,
    #     },
    # )   

    model = model_factory.build_model(
        task="segmentation", # regression       
        #backbone="terratorch_prithvi_eo_v2_600_tl",           
        #backbone="terramind_v1_base",
        backbone="terramind_v1_large",
        backbone_pretrained=True,
        backbone_modalities = ["S2L2A"],
        #backbone_drop_path = 0.1, # (?) (?) (?)                                                       
        # backbone_bands=[
        #     HLSBands.BLUE,
        #     HLSBands.GREEN,
        #     HLSBands.RED,
        #     HLSBands.RED_EDGE_1, # (?) (?)                                             
        #     HLSBands.RED_EDGE_2, # (?) (?)  
        #     HLSBands.RED_EDGE_3, # (?) (?)
        #     HLSBands.NIR_BROAD, # (?) (?)
        #     HLSBands.NIR_NARROW,
        #     HLSBands.SWIR_1,
        #     HLSBands.SWIR_2,
        # ],
        # backbone_bands=[
        #     HLSBands.BLUE,
        #     HLSBands.GREEN,
        #     HLSBands.RED,
        #     HLSBands.RED, # NIR_NARROW # RED_EDGE_1 # (?) (?)                                                           
        #     HLSBands.RED, # NIR_NARROW # RED_EDGE_2 # (?) (?)      
        #     HLSBands.NIR_NARROW, # RED_EDGE_3 # (?) (?)
        #     HLSBands.NIR_NARROW, # NIR_BROAD # (?) (?)
        #     HLSBands.NIR_NARROW,
        #     HLSBands.SWIR_1,
        #     HLSBands.SWIR_2,
        # ],
        # backbone_bands=[
        #     HLSBands.BLUE,
        #     HLSBands.GREEN,
        #     HLSBands.RED,
        #     HLSBands.RED_EDGE_1, # RED, # NIR_NARROW # RED_EDGE_1 # (?) (?)                                                                 
        #     HLSBands.RED_EDGE_2, # NIR_NARROW # RED_EDGE_2 # (?) (?)       
        #     HLSBands.RED_EDGE_3, # RED_EDGE_3 # (?) (?)
        #     HLSBands.NIR_BROAD, # NIR_BROAD # (?) (?)
        #     HLSBands.NIR_NARROW,
        #     HLSBands.SWIR_1,   
        #     HLSBands.SWIR_2, 
        # ], 
        backbone_bands = {
            "S2L2A": ["BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "SWIR_1", "SWIR_2"]
            },
        #output_bands = [HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED, HLSBands.RED_EDGE_1, HLSBands.RED_EDGE_2, HLSBands.RED_EDGE_3, HLSBands.NIR_BROAD, HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2],
        #backbone_num_frames = 1,          
        necks=[
                {
                    "name": "SelectIndices",
                    #"indices": [2, 5, 8, 11] # indices for terramind_v1_base
                    "indices": [5, 11, 17, 23] # indices for terramind_v1_large
                },
                {"name": "ReshapeTokensToImage",
                "remove_cls_token": False},  # TerraMind is trained without CLS token, which neads to be specified.
                {"name": "LearnedInterpolateToPyramidal"}  # Some decoders like UNet or UperNet expect hierarchical features. Therefore, we need to learn a upsampling for the intermediate embedding layers when using a ViT like TerraMind.
            ],
        #necks=[{"name": "SelectIndices", "indices": [7, 15, 23, 31]}, # [-1]     
        #    {"name": "ReshapeTokensToImage", "effective_time_dim": 1}, {"name": "LearnedInterpolateToPyramidal"}],
        #necks=[{"name": "SelectIndices", "indices": [7, 15, 22, 23]}, # [-1]     
        #    {"name": "ReshapeTokensToImage", "effective_time_dim": 1}, {"name": "LearnedInterpolateToPyramidal"}], 
        #decoder_channels = [512, 256, 128, 64], # 128                                                                       
        #decoder_channels = 64, # 256 # 128                   
        decoder = "UNetDecoder",
        decoder_channels = [512, 256, 128, 64],
        #decoder_scale_modules = True,    
        #head_channel_list = [128, 64], # 128 # 256 # 64                                          
        #head_channel_list = [256], # 256 # 128 # 256 # 64       
        head_dropout=0.1,
        num_classes=11, #  
        #num_classes=1, #       
        #rescale=True,  
        #loss = "dice",
        #aux_heads = [{"name": "aux_head", "decoder": "FCNDecoder", "decoder_args":"", "decoder_channels":256, "decoder_in_index":-1, "decoder_num_convs":2,"head_dropout":0.1,}],
        #aux_loss = "",
        #aux_head = 1.0,
        #ignore_index = -1,
        #freeze_backbone = False, 
    )    

    model.train()     

    from deepspeed.profiling.flops_profiler import get_model_profile         
    flops, macs, params = get_model_profile(
        model=model,
        #input_shape=(1, 3, 128, 128),  
        input_shape=(1, 10, 128, 128),
        #input_shape=(10, 10, 128, 128),
        print_profile=False, 
        module_depth=-1, # depth into the nested modules, with -1 being the inner most modules    
                                    top_modules=1, # the number of top modules to print aggregated profile
                                    warm_up=10, # the number of warm-ups before measuring the time of each module
                                    as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None
    ) 
    print(f"FLOPs: {flops}")               
    import pdb; pdb.set_trace()



    # #import numpy as np                                  
    # #import torch  
    # #import torch.nn as nn 
    # from functools import partial
    # from torch import Tensor
    # from typing import Optional

    # from timm.models.vision_transformer import _cfg 
    # from timm.models.registry import register_model
    # from timm.models.layers import trunc_normal_, lecun_normal_

    # from timm.models.layers import to_2tuple
    # from timm.models.vision_transformer import _load_weights
    # from einops import rearrange
    # import math
    # #from utils_mamba2.pos_embed import *
    # from collections import namedtuple

    # from mamba_simple import Mamba
    # from mamba_ssm.utils.generation import GenerationMixin
    # from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
    # try:
    #     from mamba_ssm.ops.triton.layernorm import  layer_norm_fn, rms_norm_fn  #RMSNorm,
    # except ImportError:
    #     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

    # import torch.nn as nn
    # from timm.models.layers import DropPath
    # from einops import rearrange
    # class RMSNorm(torch.nn.Module):

    #     def __init__(self, hidden_size, eps=1e-5, dropout_p=0.0, device=None, dtype=None):
    #         factory_kwargs = {"device": device, "dtype": dtype}
    #         super().__init__()
    #         self.eps = eps
    #         if dropout_p > 0.0:
    #             self.drop = torch.nn.Dropout(dropout_p)
    #         else:
    #             self.drop = None
    #         self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
    #         self.register_parameter("bias", None)
    #         self.reset_parameters()

    #     def reset_parameters(self):
    #         torch.nn.init.ones_(self.weight)

    #     def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
    #         return rms_norm_fn(
    #             x,
    #             self.weight,
    #             self.bias,
    #             residual=residual,
    #             eps=self.eps,
    #             dropout_p=self.drop.p if self.drop is not None and self.training else 0.0,
    #             prenorm=prenorm,
    #             residual_in_fp32=residual_in_fp32,
    #         )

    # class Mlp(nn.Module):
    #     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
    #         super().__init__()
    #         out_features = out_features or in_features
    #         hidden_features = hidden_features or in_features
    #         self.fc1 = nn.Linear(in_features, hidden_features)
    #         self.act = act_layer()
    #         self.fc2 = nn.Linear(hidden_features, out_features)
    #         self.drop = nn.Dropout(drop)

    #     def forward(self, x):
    #         x = self.fc1(x)
    #         x = self.act(x)
    #         x = self.drop(x)
    #         x = self.fc2(x)
    #         x = self.drop(x)
    #         return x

    # class CrossAttention(nn.Module):
    #     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
    #         super().__init__()
    #         self.num_heads = num_heads
    #         head_dim = dim // num_heads
    #         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
    #         self.scale = qk_scale or head_dim ** -0.5

    #         self.q = nn.Linear(dim, dim, bias=qkv_bias)
    #         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
    #         self.attn_drop = nn.Dropout(attn_drop)
    #         self.proj = nn.Linear(dim, dim)
    #         self.proj_drop = nn.Dropout(proj_drop)

    #     def forward(self, q, kv, mask):
    #         B, N, C = q.shape
    #         q = self.q(q).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    #         kv = self.kv(kv).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    #         q, k, v = q[0], kv[0], kv[1]   # make torchscript happy (cannot use tensor as tuple)
    #         attn = (q @ k.transpose(-2, -1)) * self.scale
    #         attn += mask
    #         attn = attn.softmax(dim=-1)
    #         attn = self.attn_drop(attn)
    #         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    #         x = self.proj(x)
    #         x = self.proj_drop(x)
    #         return x

    # class DecoderBlock(nn.Module):
    #     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
    #                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    #         super().__init__()
    #         self. attn2 = CrossAttention(
    #             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
    #         self.norm2_1 = norm_layer(dim)
    #         self.norm2_2 = norm_layer(dim)
    #         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
    #         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    #         self.norm2 = norm_layer(dim)
    #         mlp_hidden_dim = int(dim * mlp_ratio)
    #         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    #     def forward(self, q, kv, mask):
    #         q = q + self.attn2(self.norm2_1(q), self.norm2_2(kv), mask)
    #         q = q + self.mlp(self.norm2(q))
    #         return q

    # class PatchEmbed(nn.Module):
    #     """ 2D Image to Patch Embedding
    #     """
    #     def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None,
    #                 flatten=True):
    #         super().__init__()
    #         img_size = to_2tuple(img_size)
    #         patch_size = to_2tuple(patch_size)
    #         self.img_size = img_size
    #         self.patch_size = patch_size
    #         self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
    #         self.num_patches = self.grid_size[0] * self.grid_size[1]
    #         self.flatten = flatten

    #         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
    #         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    #     def forward(self, x):
    #         B, C, H, W = x.shape
    #         assert H == self.img_size[0] and W == self.img_size[1], \
    #             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
    #         x = self.proj(x)
    #         if self.flatten:
    #             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
    #         x = self.norm(x)
    #         return x

    # class SwiGLU(nn.Module):
    #     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
    #                 norm_layer=nn.LayerNorm, subln=False
    #                 ):
    #         super().__init__()
    #         out_features = out_features or in_features
    #         hidden_features = hidden_features or in_features

    #         self.w1 = nn.Linear(in_features, hidden_features)
    #         self.w2 = nn.Linear(in_features, hidden_features)

    #         self.act = act_layer()
    #         self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
    #         self.w3 = nn.Linear(hidden_features, out_features)

    #         self.drop = nn.Dropout(drop)

    #     def forward(self, x):
    #         x1 = self.w1(x)
    #         x2 = self.w2(x)
    #         hidden = self.act(x1) * x2
    #         x = self.ffn_ln(hidden)
    #         x = self.w3(x)
    #         x = self.drop(x)
    #         return x

    # class Block(nn.Module):
    #     def __init__(
    #             self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    #     ):
    #         super().__init__()
    #         self.residual_in_fp32 = residual_in_fp32
    #         self.fused_add_norm = fused_add_norm
    #         self.mixer = mixer_cls(dim)
    #         self.mlp = SwiGLU(dim, dim * 4 * 2 // 3)
    #         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    #         self.norm1 = nn.LayerNorm(dim)
    #         self.norm2 = nn.LayerNorm(dim)

    #     def forward(
    #             self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    #     ):
    #         hidden_states = hidden_states + self.drop_path(
    #             self.mixer(self.norm1(hidden_states), inference_params=inference_params))
    #         hidden_states = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))

    #         return hidden_states

    #     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
    #         return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    # def create_block(
    #         d_model,
    #         ssm_cfg=None,
    #         norm_epsilon=1e-5,
    #         drop_path=0.,
    #         rms_norm=False,
    #         residual_in_fp32=False,
    #         fused_add_norm=False,
    #         layer_idx=None,
    #         device=None,
    #         dtype=None,
    #         bimamba_type="none",
    #         if_devide_out=False,
    #         init_layer_scale=None,
    # ):
    #     if ssm_cfg is None:
    #         ssm_cfg = {}
    #     factory_kwargs = {"device": device, "dtype": dtype}
    #     mixer_cls = partial(Mamba, expand=1, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out,
    #                         init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    #     norm_cls = partial(
    #         nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    #     )
    #     block = Block(
    #         d_model,
    #         mixer_cls,
    #         norm_cls=norm_cls,
    #         drop_path=drop_path,
    #         fused_add_norm=fused_add_norm,
    #         residual_in_fp32=residual_in_fp32,
    #     )
    #     block.layer_idx = layer_idx
    #     return block

    # # https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
    # def _init_weights(
    #         module,
    #         n_layer,
    #         initializer_range=0.02,  # Now only used for embedding layer.
    #         rescale_prenorm_residual=True,
    #         n_residuals_per_layer=1,  # Change to 2 if we have MLP
    # ):
    #     if isinstance(module, nn.Linear):
    #         if module.bias is not None:
    #             if not getattr(module.bias, "_no_reinit", False):
    #                 nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         nn.init.normal_(module.weight, std=initializer_range)

    #     if rescale_prenorm_residual:
    #         for name, p in module.named_parameters():
    #             if name in ["out_proj.weight", "fc2.weight"]:
    #                 nn.init.kaiming_uniform_(p, a=math.sqrt(5))
    #                 with torch.no_grad():
    #                     p /= math.sqrt(n_residuals_per_layer * n_layer)

    # def segm_init_weights(m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=0.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Conv2d):
    #         # NOTE conv was left to pytorch default in my original init
    #         lecun_normal_(m.weight)
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)
    #     elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
    #         nn.init.zeros_(m.bias)
    #         nn.init.ones_(m.weight)

    # class VisionMamba(nn.Module):
    #     def __init__(self,
    #                 img_size=224,
    #                 patch_size=16,
    #                 stride=16,
    #                 depth=24,
    #                 embed_dim=192,
    #                 dec_embed_dim=192,
    #                 channels=3,
    #                 num_classes=1000,
    #                 ssm_cfg=None,
    #                 drop_rate=0.,
    #                 drop_path_rate=0.1,
    #                 norm_epsilon: float = 1e-5,
    #                 rms_norm: bool = False,
    #                 initializer_cfg=None,
    #                 fused_add_norm=False,
    #                 residual_in_fp32=False,
    #                 device=None,
    #                 dtype=None,
    #                 if_bidirectional=False,
    #                 if_abs_pos_embed=False,
    #                 bimamba_type="none",
    #                 if_devide_out=False,
    #                 init_layer_scale=None,
    #                 crop_size=96,
    #                 **kwargs):
    #         factory_kwargs = {"device": device, "dtype": dtype} 
    #         # add factory_kwargs into kwargs 
    #         kwargs.update(factory_kwargs)
    #         super().__init__()
    #         self.residual_in_fp32 = residual_in_fp32
    #         self.fused_add_norm = fused_add_norm
    #         self.if_bidirectional = if_bidirectional
    #         self.if_abs_pos_embed = if_abs_pos_embed
    #         self.crop_size = crop_size
    #         self.window_size = self.crop_size //patch_size
    #         if depth==12:
    #             self.skip = [6, 8, 10, 12]
    #         elif depth==24:
    #             self.skip = [12, 16, 20, 24]

    #         # pretrain parameters
    #         self.num_classes = num_classes
    #         self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

    #         self.patch_embed = PatchEmbed(
    #             img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
    #         num_patches = self.patch_embed.num_patches
    #         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim), requires_grad=False)
    #         self.angle_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
    #         self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
    #         # transformer blocks
    #         self.layers = nn.ModuleList(
    #             [
    #                 create_block(
    #                     embed_dim,
    #                     ssm_cfg=ssm_cfg,
    #                     norm_epsilon=norm_epsilon,
    #                     rms_norm=rms_norm,
    #                     residual_in_fp32=residual_in_fp32,
    #                     fused_add_norm=fused_add_norm,
    #                     layer_idx=i,
    #                     bimamba_type=bimamba_type,
    #                     drop_path=0.,
    #                     if_devide_out=if_devide_out,
    #                     init_layer_scale=init_layer_scale,
    #                     **factory_kwargs,
    #                 )
    #                 for i in range(depth)
    #             ]
    #         )
    #         self.dec_embed_dim = dec_embed_dim
    #         self.ar_token = nn.Parameter(torch.zeros(1, 1, self.dec_embed_dim))
    #         self.dec_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.dec_embed_dim), requires_grad=False)
    #         self.pos_drop = nn.Dropout(p=drop_rate)
    #         self.enc2dec = nn.Linear(embed_dim * 4, self.dec_embed_dim * 4)
    #         self.dec_block = nn.ModuleList([
    #             DecoderBlock(self.dec_embed_dim, self.dec_embed_dim // 64, 4, qkv_bias=True, qk_scale=None,
    #                         norm_layer=nn.LayerNorm)
    #             for i in range(4)])
    #         self.norm_1 = nn.LayerNorm(embed_dim)
    #         self.norm_2 = nn.LayerNorm(embed_dim)
    #         self.norm_3 = nn.LayerNorm(embed_dim)
    #         self.norm_4 = nn.LayerNorm(embed_dim)
    #         self.ar_norm = nn.LayerNorm(self.dec_embed_dim)
    #         self.ar_pred = nn.Linear(self.dec_embed_dim, 768)

    #         # original init
    #         self.patch_embed.apply(segm_init_weights)
    #         if if_abs_pos_embed:
    #             pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=False)
    #             self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    #             dec_pos_embed = get_2d_sincos_pos_embed(self.dec_pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
    #                                                 cls_token=False)

    #             self.dec_pos_embed.data.copy_(torch.from_numpy(dec_pos_embed).float().unsqueeze(0))
    #         trunc_normal_(self.ar_token, std=.02)
    #         trunc_normal_(self.angle_embed, std=.02)
    #         # mamba init
    #         self.apply(
    #             partial(
    #                 _init_weights,
    #                 n_layer=depth,
    #                 **(initializer_cfg if initializer_cfg is not None else {}),
    #             )
    #         )
    #         self.dec_block.apply(self.atten_init_weights)
    #         # 196
    #         self.register_buffer("mask", self.mask_generate(9-1, 16))
    #     def mask_generate(self, segment, tokens_per_segment):
    #         mask = torch.tril(torch.ones((segment, segment), dtype=torch.float))
    #         mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)
    #         mask = torch.repeat_interleave(mask, repeats=tokens_per_segment, dim=0)
    #         mask = torch.repeat_interleave(mask, repeats=tokens_per_segment, dim=1)

    #         return mask
    #     def atten_init_weights(self, m):
    #         if isinstance(m, nn.Linear):
    #             torch.nn.init.xavier_uniform_(m.weight)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)

    #     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
    #         return {
    #             i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    #             for i, layer in enumerate(self.layers)
    #         }

    #     @torch.jit.ignore
    #     def no_weight_decay(self):
    #         return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    #     @torch.jit.ignore()
    #     def load_pretrained(self, checkpoint_path, prefix=""):
    #         _load_weights(self, checkpoint_path, prefix)

    #     def calculate_rotated_indices(self, top_start, left_start, image_width):
    #         window_indices = []
    #         for i in range(self.window_size):
    #             for j in range(self.window_size):
    #                 idx = (top_start + i) * image_width + (left_start + j)
    #                 window_indices.append(idx)

    #         return window_indices

    #     def get_rotated_patch_indices(self, top_starts, left_starts, window_number, image_width):
    #         rotated_patch_indices = []

    #         for i in range(top_starts.shape[0]):
    #             window_indices = []
    #             for j in range(window_number):
    #                 top_start = top_starts[i, j]
    #                 left_start = left_starts[i, j]
    #                 rotated_indices = self.calculate_rotated_indices(top_start, left_start, image_width)
    #                 window_indices.append(rotated_indices)

    #             rotated_patch_indices.append(window_indices)

    #         return torch.tensor(rotated_patch_indices, device=top_starts.device)

    #     def forward_features(self, x, inference_params,x_starts, y_starts,crop_size):
    #         x = self.patch_embed(x)
    #         B, N, C = x.shape
    #         H, W = int(np.sqrt(N)), int(np.sqrt(N))
    #         x = x + self.pos_embed
    #         x = self.pos_drop(x)
    #         rotation_mask = torch.zeros(B, H * W, device=x.device, dtype=torch.bool)
    #         for batch_idx in range(B):
    #             current_crop_size = crop_size[batch_idx]
    #             current_x_starts = x_starts[batch_idx]
    #             current_y_starts = y_starts[batch_idx]
    #             token_per_pixel=16
    #             x_start_token = (current_x_starts / token_per_pixel).long()
    #             y_start_token = (current_y_starts / token_per_pixel).long()
    #             crop_span = (current_crop_size / token_per_pixel).long()
    #             x_indices = torch.arange(x_start_token.item(), x_start_token.item() + crop_span.item(), device=x.device, dtype=torch.long)
    #             y_indices = torch.arange(y_start_token.item(), y_start_token.item() + crop_span.item(), device=x.device,dtype=torch.long)
    #             grid_indices = y_indices * W + x_indices
    #             rotation_mask[batch_idx, grid_indices] = True

    #         angle_embed_expanded = self.angle_embed.expand(B, H * W, C)
    #         x = x + angle_embed_expanded * rotation_mask.unsqueeze(-1)
    #         H,W = int(np.sqrt(N)), int(np.sqrt(N))
    #         x = x.reshape(B, H, W, C)
    #         x = rearrange(x, "b (h p1) (w p2) c -> b (h w) (p1 p2) c", p1=4, p2=4)
    #         hidden_states = x[:, :-1].reshape(B, -1, C)
    #         features = []
    #         count=0
    #         for layer in self.layers:

    #             hidden_states = layer(
    #                 hidden_states, inference_params=inference_params
    #             )
    #             count += 1
    #             if count in self.skip:
    #                 features.append(hidden_states)

    #         features = [self.norm_1(features[0]), self.norm_2(features[1]),
    #                     self.norm_3(features[2]),self.norm_4(features[3])]
    #         features = self.enc2dec(torch.cat(features, dim=-1))
    #         B, N, C = features.shape
    #         features = features.reshape(B, 8, N//8, C)
    #         features = features.reshape(B, -1, C)
    #         B, N, C = features.shape
    #         assert N==16*8
    #         return features.reshape(B, N, C//4, 4)

    #     def forward_decoder(self, latent_ar, decoder_pos_embed):
    #         B, N, C, depth = latent_ar.shape
    #         ar_token = self.ar_token + decoder_pos_embed
    #         H, W = int(np.sqrt(ar_token.shape[1])), int(np.sqrt(ar_token.shape[1]))
    #         ar_token = ar_token.reshape(ar_token.shape[0], H, W, C)
    #         ar_token = rearrange(ar_token, "b (h p1) (w p2) c -> b (h w) (p1 p2) c", p1=4, p2=4)
    #         ar_token = ar_token[:, 1:].reshape(1, -1, C)
    #         ar_token = ar_token.repeat(B,1,1)
    #         count = 0
    #         for blk in self.dec_block:
    #             ar_token = blk(ar_token, latent_ar[:, :, :, count],self.mask)
    #             count += 1
    #         ar_token = self.ar_norm(ar_token)
    #         ar_token = self.ar_pred(ar_token)
    #         return ar_token

    #     def patchify(self, imgs):
    #         p = self.patch_embed.patch_size[0]
    #         assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    #         h = w = imgs.shape[2] // p
    #         x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    #         x = torch.einsum('nchpwq->nhwpqc', x)
    #         x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    #         return x

    #     def forward_loss(self, imgs, pred):
    #         target = self.patchify(imgs)
    #         if True:
    #             mean = target.mean(dim=-1, keepdim=True)
    #             var = target.var(dim=-1, keepdim=True)
    #             target = (target - mean) / (var + 1.e-6) ** .5
    #         B, N, C = target.shape
    #         H, W = int(np.sqrt(N)), int(np.sqrt(N))
    #         target = target.reshape(B, H, W, C)
    #         target = rearrange(target, "b (h p1) (w p2) c -> b (h w) (p1 p2) c", p1=4, p2=4)
    #         target = target[:, 1:].reshape(B, -1, C)
    #         loss = (pred - target) ** 2
    #         return loss
    #     def forward_loss_simple(self, target, pred):
    #         loss = (pred - target) ** 2
    #         return loss

    #     def process_labels(self, imgs, block_size):
    #         target = self.patchify(imgs)  # # [batch_size, num_patches, channels * patch_size * patch_size] 
    #         mean = target.mean(dim=-1, keepdim=True)
    #         var = target.var(dim=-1, keepdim=True)
    #         target = (target - mean) / (var + 1.e-6) ** 0.5
    #         B, N, C = target.shape
    #         H, W = int(math.sqrt(N)), int(math.sqrt(N))
    #         target = target.reshape(B, H, W, C)
    #         target = rearrange(target, f"b (h p1) (w p2) c -> b (h w) (p1 p2) c", p1=block_size,
    #                         p2=block_size)
    #         patches_per_block = block_size ** 2
    #         num_patches_to_remove = 16
    #         num_blocks_to_remove = (num_patches_to_remove + patches_per_block - 1) // patches_per_block
    #         target = target[:, num_blocks_to_remove:, :]
    #         target = target.reshape(B, target.shape[1], -1)
    #         return target

    #     def concat_block_embeds(self, x, block_size):
    #         bs, num_patches, dim = x.shape
    #         pad_tokens = torch.zeros(bs, 16, dim, device=x.device, dtype=x.dtype)
    #         x_padded = torch.cat([pad_tokens, x], dim=1)
    #         x_padded = x_padded.transpose(1, 2)
    #         grid_size = int(math.sqrt(144))
    #         assert grid_size % block_size == 0, f"block_size {block_size} 必须能整除 grid_size {grid_size}"
    #         x_padded = x_padded.view(bs, dim, grid_size, grid_size)
    #         patches = x_padded.unfold(2, block_size, block_size).unfold(3, block_size,block_size)
    #         patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(bs, -1, dim, block_size,block_size)
    #         x_concat = patches.flatten(2)
    #         patches_per_block = block_size ** 2
    #         num_patches_to_remove = 16
    #         num_blocks_to_remove = (num_patches_to_remove + patches_per_block - 1) // patches_per_block
    #         x_concat = x_concat[:, num_blocks_to_remove:, :]
    #         return x_concat

    #     def forward(self, x, ori_imgs, tops, lefts,crop_size,inference_params=None): 
    #         labels = ori_imgs  
    #         x = self.forward_features(x, inference_params, tops, lefts,crop_size) 
    #         x = self.forward_decoder(x, self.dec_pos_embed)
    #         block_sizes = [6]
    #         total_loss = 0.0
    #         for block_size in block_sizes:
    #             num_patches = x.shape[1]
    #             grid_size = int(math.sqrt(num_patches))
    #             pred_block = self.concat_block_embeds(x, block_size)
    #             target_block = self.process_labels(labels, block_size)
    #             loss = self.forward_loss_simple(target_block, pred_block)
    #             total_loss += loss.mean() 
    #         loss = self.forward_loss(labels, x)     
    #         total_loss=loss+0.1*total_loss 
    #         return total_loss  

    # model = VisionMamba(
    #     channels=10, patch_size=16, img_size=128, embed_dim=192, depth=12, dec_embed_dim=512, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
    #     if_abs_pos_embed=True, bimamba_type="None")                                   
    
    print(f'Training on: {model_name}')                         
    print('--'*10)    
    
    #data_parallel = True  
    
    if data_parallel:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!") 
            # # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs           
            model = nn.DataParallel(model, device_ids=device_ids)
 
    model = nn.DataParallel(model)
    model.to(model_device)
    
    # if model_name == 'SatMAE' or model_name =='SatMAE_classifier':
    #     model_summary = summary(model,
    #                             input_size=(batch_size, input_channels, 96, 96), )

    # elif model_name == 'prithvi' or model_name =='prithvi_classifier':
    #     model_summary = summary(model,
    #                             input_size=(batch_size, 6, 224, 224), dtypes=[torch.float32])

    # elif model_name in ['seasonal_contrast', 'resnet_imagenet', 'resnet', 'seasonal_contrast_classifier']:
    #     model_summary = summary(model,
    #                             input_size=(batch_size, input_channels, 224, 224), )

    # else:
    #     model_summary = summary(model, input_size=(batch_size, input_channels, input_size, input_size))

    trainer = get_trainer(model_name, downstream_task, epochs, lr, model, model_device, lr_scheduler, warmup, early_stop, dl_train,
                          dl_val, dl_test, NAME, OUTPUT_FOLDER, vis_val, warmp_steps, warmup_gamma)

    model.train()   
    
    trainer.train()                         
    trainer.test()    
    
    # trainer.save_info(model_summary=model_summary, n_shot=n_shot, p_split=split_ratio, warmup=warmup,
    #                   lr=init_lr)                                                                                                                                           

if __name__ == "__main__":
    parser, parser_yaml = get_args()
    args_yaml, remainder = parser_yaml.parse_known_args()
    
    if args_yaml.read_yaml is not None:
        print(f"WARNING: overwriting all parameters with defaults stored in {args_yaml.read_yaml}")
        args = read_yaml(args_yaml.read_yaml)
    else:
        args = parser.parse_args()

    main(**vars(args))

