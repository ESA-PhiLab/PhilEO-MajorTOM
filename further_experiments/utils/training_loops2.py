# # Standard Library
import os

from tqdm import tqdm
from matplotlib import pyplot as plt

import matplotlib
# import PyQt5
# matplotlib.use('QtAgg')

# # PyTorch                        
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from torchvision import transforms

import json
import numpy as np

# # utils      
from utils import visualize
from utils import config_lc
from pytorch_msssim import ms_ssim

import segmentation_models_pytorch as smp 

import glob       
import math             

class TrainBase():        
    def __init__(self, model: nn.Module, device: torch.device, train_loader: DataLoader, val_loader: DataLoader,
                 test_loader: DataLoader, epochs:int = 50, early_stop:int=25, lr: float = 0.001, lr_scheduler: str = None, warmup:bool=True,
                 metrics: list = None, name: str="model", out_folder :str ="trained_models/", visualise_validation:bool=True, 
                 warmup_steps:int=5, warmup_gamma:int=10):

        self.test_loss = None 
        self.last_epoch = None
        self.best_sd = None
        self.epochs = epochs
        self.early_stop = early_stop
        self.learning_rate = lr
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.metrics = metrics
        self.lr_scheduler = lr_scheduler
        self.warmup = warmup
        self.warmup_steps = warmup_steps
        self.name = name
        self.out_folder = out_folder
        self.visualise_validation = visualise_validation
        if visualise_validation:
            os.makedirs(f'{self.out_folder}/val_images', exist_ok=True)

        self.scaler, self.optimizer = self.set_optimizer()
        self.criterion = self.set_criterion()
        self.scheduler = self.set_scheduler()

        if self.warmup:
            multistep_milestone =  list(range(1, self.warmup_steps+1))
            self.scheduler_warmup = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=multistep_milestone, gamma=(warmup_gamma))

        # # initialize torch device 
        #torch.set_default_device(self.device)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA available.") 
        else:
            print("No CUDA device available.")

        # init useful variables
        self.best_epoch = 0
        self.best_loss = None
        self.best_model_state = model.state_dict().copy()
        self.epochs_no_improve = 0

        # used for plots  
        self.tl = []
        self.vl = []
        self.e = []
        self.lr = []

    def set_optimizer(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.learning_rate, eps=1e-06)

        scaler = GradScaler()  

        # # Save the initial learning rate in optimizer's param_groups                                                                                                              
        for param_group in optimizer.param_groups:     
            param_group['initial_lr'] = self.learning_rate

        return scaler, optimizer  

    def set_criterion(self):  
        return nn.MSELoss()   
        #return nn.CrossEntropyLoss()

    def set_scheduler(self):
        if self.lr_scheduler == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                20,
                2,
                eta_min=0.000001,
                last_epoch=self.epochs - 1,
            )
        elif self.lr_scheduler == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=6, min_lr=1e-6)
        else:
            scheduler = None
        return scheduler

    def get_loss(self, images, labels):
        #outputs = self.model(images)        

        # # # Create patches and apply flatten operation          
        # # # With the batch support the operation becomes this in Torch 
        # # # Credit: # # https://discuss.pytorch.org/t/how-to-extract-patches-from-an-image/79923/4
        # #unfolded_batch = image_batch.unfold(1, C, C).unfold(2, P, P).unfold(3, P, P) 
        # C = 10   
        # P = 16 
        # B = 100
        # desired_image_size = 128
        # patch_dim = int(P * P * C)
        # number_of_patches = int((desired_image_size * desired_image_size)/ (P * P))
        # unfolded_batch = images.unfold(1, C, C).unfold(2, P, P).unfold(3, P, P) 
        # flattened_patches = unfolded_batch.contiguous().view(B, -1, C * P * P).float()

        # # # Linear Projection process    
        # #patch_weights = nn.Parameter(torch.empty(P * P * C, D).normal_(std=0.02)) 
        # D = 768   
        # patch_weights = nn.Parameter(torch.empty(P * P * C, D).normal_(std=0.02)).cuda()

        # patch_embeddings = torch.matmul(flattened_patches, patch_weights)
        
        # # # Initialisation of Class Token     
        # #class_token = nn.Parameter(torch.empty(1, 1, D).normal_(std=0.02)) 
        # class_token = nn.Parameter(torch.empty(1, 1, D).normal_(std=0.02)).cuda() 

        # batch_class_token = class_token.expand(B, -1, -1)
        # patch_embeddings_with_class_token = torch.cat([batch_class_token, patch_embeddings], dim=1)

        # # # Addition of the Positional Embeddings to Patch Embeddings with Class Tokens   
        # #positional_embedding = nn.Parameter(torch.empty(B, number_of_patches + 1, D).normal_(std=0.02))
        # positional_embedding = nn.Parameter(torch.empty(B, number_of_patches + 1, D).normal_(std=0.02)).cuda() 

        # embeddings = patch_embeddings_with_class_token + positional_embedding

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
        
        # attention_values = self_attention(embeddings)

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
        # multi_head_attention_score = multi_head_attention(patch_embeddings_with_class_token)

        # # # torch.Size([100, 65, 768])   

        # #print(f'Shape of the Multi-Head Attention: {multi_head_attention_score.shape}')
        # #assert multi_head_attention_score.shape == (B, number_of_patches + 1, D)



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
        
        # output = mlp(multi_head_attention_score) 

        # # # torch.Size([100, 65, 768])    

        # #assert output.shape == (B, number_of_patches + 1, D) 
        # #print(F'Shape of MLP output: {output.shape}')



        # #print(output) 
        # #print(output.shape)

        # # # torch.Size([100, 65, 768])   






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
        #         # Applying dropout  
        #         dropout_embeddings = self.dropout1(embeddings)
        #         # Layer normalization
        #         normalized_embeddings = self.layer_norm1(dropout_embeddings)
        #         # Calculation of multi-head attention
        #         attention_scores = self.multi_head_attention(normalized_embeddings)
        #         # Applying the second dropout
        #         dropout_attention_scores = self.dropout2(attention_scores)
        #         # Residual connection with second dropout output and initial input
        #         residuals_embeddings = embeddings + dropout_attention_scores
        #         # apply layer normalization
        #         normalized_residuals_embeddings = self.layer_norm2(residuals_embeddings)
        #         # aply MLP 
        #         transformed_results = self.mlp(normalized_residuals_embeddings)
        #         # Applying the third dropout
        #         dropout_transformed_results = self.dropout3(transformed_results)
        #         # Residual connection with last dropout output and first residual output
        #         output = residuals_embeddings + dropout_transformed_results

        #         return output



        # # # init transformer encoder         
        # transformer_encoder = TransformerEncoder(embedding_dim=D, num_heads=num_heads, hidden_dim=hidden_dim, dropout=0.1) 

        # # # compute transformer encoder output
        # output = transformer_encoder(embeddings)

        # # # torch.Size([100, 65, 768])   
        


        # #print(f'Shape of the output of Transformer Encoders: {output.shape}')
        # #assert output.shape == (B, number_of_patches + 1, D)



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



        
        # # # Classifier "token" as used by standard language architectures     
        # class_token_output = output[:, 0]    

        # # # initialise number of classes  
        # #n_class = 10
        # n_class = 11

        
        
        # # # initialise classification head   
        # mlp_head = MLPHead(D, n_class) 

        # cls_output = mlp_head(class_token_output)

        
        
        # #print(cls_output.shape)
        # #sadfasdf

        
        
        # # # torch.Size([100, 10])   



        # # # size of output    
        # #print(f'Shape of the MLP Head output: {cls_output.shape}')
        # #assert list(cls_output.shape) == [B, n_class]
        
        
        
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

        #         # mlp head 
        #         #self.mlp_head = MLPHead(embedding_dim, num_classes)
        #         self.mlp_head = PixelMLPHead(embedding_dim, num_classes)

        #     def forward(self, images): 
        #         # # get patch size and channel size        
        #         P, C = self.patch_size, self.C 

        #         #print(images.shape) 
        #         #sadfasdf

        #         # # torch.Size([100, 10, 128, 128])  



        #         # create image patches
        #         patches = images.unfold(1, C, C).unfold(2, P, P).unfold(3, P, P).contiguous().view(images.size(0), -1, C * P * P).float()

        #         #print(patches.shape) 
        #         #asdfasdf
                
        #         # # torch.Size([100, 64, 2560])  
        #         # # (B, (image_size/patchsize)*(image_size/patchsize), C*patchsize*patchsize) 
                


        #         # patch embeddings
        #         patch_embeddings = torch.matmul(patches , self.W)

        #         #print(patch_embeddings.shape)
        #         #sadfsadf

        #         # # torch.Size([100, 64, 768])  



        #         # class token + patch_embeddings
        #         batch_class_token = self.class_tokens.expand(patch_embeddings.shape[0], -1, -1)
        #         patch_embeddings_with_class_token = torch.cat([batch_class_token, patch_embeddings], dim=1)

        #         # add positional embedding
        #         embeddings = patch_embeddings_with_class_token + self.positional_embeddings

        #         #print(embeddings.shape)
        #         #sadfasdfs

        #         # # torch.Size([100, 65, 768])   



        #         # execute Transformer encoders 
        #         transformer_encoder_output = self.transformer_encoder(embeddings)

        #         # Classifier "token" as used by standard language architectures
        #         output_class_token = transformer_encoder_output[:, 0]

        #         return self.mlp_head(output_class_token)


        
        # # # init vision transformer model          
        # num_layers = 12   
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

        
        
        # # we can use image_batch as it is    
        # #vit_output = vision_transformer(images) 
        # output = model(images) 

        # # # torch.Size([100, 10])       



        # #assert vit_output.size(dim=1) == n_class  
        # #print(vit_output.shape) 



        # #print(vit_output.shape)  

        # # # torch.Size([100, 11, 128, 128])   

        # # # torch.Size([100, 10, 128, 128])  
        # # # (B, numclasses, W, H)    



        # asdfzsd




















        outputs = self.model(images)
        
        # print(outputs)
        
        # print(outputs.shape)

        # asdfsadf
        

        
        #outputs, outputs2 = self.model(images)  
        
        
        
        #print('')     
        #print(outputs.shape)
        #print(outputs2.shape)

        #print(images.shape)

        #print(labels.shape)

        #sadfsadf



        
        
        
        # #asdfasd
        # #outputs = outputs.argmax(axis=1).flatten()                
        # outputs = outputs.flatten()
        # #labels = labels.squeeze().flatten()
        
        

        
        
        # # #outputs = self.model(images)         
        # # #outputs = outputs.argmax(axis=1).flatten()          
        # # #labels = labels.squeeze().flatten()
        
        # # #outputs = outputs.argmax(axis=1).flatten()      
        
        # # #outputs = outputs.argmax(axis=1).flatten()

        # # outputs, outputs2 =  outputs.argmax(axis=1).flatten(), outputs.max(axis=1)[0].flatten()

        # # # print(outputs)        
        # # # print(outputs.shape)  
        # # # sadfasdfas

        
        
        
        
        
        
        
        # labels = labels.squeeze().flatten()                     
        # #sadfasdfas 







        #print(outputs.shape)     
        #print(labels.shape)
        #sadfsadf
        
        
        
        #outputs = outputs.output   
        
        
        
        loss = self.criterion(outputs, labels)

        





                        
                


        # # #print(loss.shape)  

        # # #print(((outputs - labels) ** 2).shape)     

        # # #print(((outputs - labels) ** 2).mean())  

        # # #print((((outputs - labels) ** 2).flatten()).shape)



        # # # #outputs = [4, 5, 2]        
        # # # import torch
        # # # #outputs = torch.tensor([4, 5, 2, 14])
        # # # outputs = torch.tensor([4, 5, 2])

        # # # #labels = [4, 1, 0]                   
        # # # #labels = torch.tensor([4, 1, 0, 2])
        # # # labels = torch.tensor([4, 1, 0])

        # # # #loss = 0.           





        # # #a, _ = ((outputs - labels) ** 2).flatten().sort() 
        # # #a, _ = ((outputs - labels) ** 2).flatten().sort(descending=False) 
        # # a, _ = ((outputs - labels) ** 2).flatten().sort(descending=True) 

        # # #print(a) 

        # # #sadfasdf

        # # #print(len(((outputs - labels) ** 2).flatten())); print(len(a)) #a = a[0.2*len(((outputs - labels) ** 2).flatten())]  
        # # #sadfs
        # # #a = a[0.8*len(((outputs - labels) ** 2).flatten())] 
        # # #a = a[round(0.2*len(((outputs - labels) ** 2).flatten()))]  
        # # #a = a[round(0.2*len(((outputs - labels) ** 2).flatten()))]  
        # # a = a[round(0.8*len(((outputs - labels) ** 2).flatten()))]  
        # # #a = a[(0.8*len(((outputs - labels) ** 2).flatten())).floor()]  

        # # #print(a)

        # # #asdfas

        # # #a = a[round(0.8*len(((outputs - labels) ** 2).flatten()))]  
        # # #print((torch.where((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)>a, torch.ones_like((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)), 0.*(((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape))).shape)
        # # #labels2new = torch.where((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)>a, torch.ones_like((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)), 0.*(((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape))
        # # labels2new = torch.where((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)>a, torch.zeros_like((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)), torch.ones_like((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)))
        # # #labels2new = torch.where((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)>=a, torch.zeros_like((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)), torch.ones_like((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)))
        # # # # torch.where(condition, input, other)      

        # # #print(((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)).shape) 

        # # #print(labels2new)       
        # # #sadfasdf

        # # #print(torch.count_nonzero(labels2new))  
        # # #print(torch.count_nonzero(labels2new-1.))  
        # # #sadfkzs



        # # #print(torch.count_nonzero(labels2new))  
        # # #print(torch.count_nonzero(labels2new-1.))  
        # # #sadfkzs



        # # #arr1inds = arr1.argsort()
        # # #sorted_arr1 = arr1[arr1inds[::-1]]
        # # #sorted_arr2 = arr2[arr1inds[::-1]]

        # # arr1inds = ((outputs - labels) ** 2).flatten().argsort(descending=True) 
        # # #sorted_arr1 = arr1[arr1inds[::-1]]  
        # # #sorted_arr2 = arr2[arr1inds[::-1]]
        # # #sorted_arr = outputs[arr1inds]
        # # #sorted_arr = outputs[arr1inds]
        # # #sorted_arr = outputs[arr1inds]
        # # #sorted_arr = outputs2[arr1inds]
        # # #sorted_arr = outputs2.flatten()[arr1inds].view(outputs.shape)
        # # sorted_arr = outputs2.flatten()[arr1inds].unflatten(dim=0, sizes=outputs.shape)

        # # #print(sorted_arr)  
        # # #asdfasdf

        # # #print(outputs)
        # # #asdfzsdf



        # # #print(sorted_arr)    
        # # #print(labels2new) 
        # # #sadfasdfas









        # # #print('')                                                                                                       

        # # #print((outputs - labels) ** 2)    

        # # #print('') 

        # # #print((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)) 

        # # #sadfsadf



        # # #loss +=                                                                                                                                                                                                                                                                                                                                                           
        # # #loss += self.criterion(outputs2, labels2new)                      
        # # #loss += self.criterion(outputs2, labels2new)  
        # # #loss += 0.1 * self.criterion(outputs2, labels2new)  
        # # #loss += self.criterion(outputs2, labels2new) 
        # # #loss += self.criterion(outputs2, labels2new)
        # # #loss += self.criterion(outputs2, labels2new)
        # # #loss += self.criterion(sorted_arr, labels2new)
        # # loss += 0.1 * self.criterion(sorted_arr, labels2new)











        # #a, _ = ((outputs - labels) ** 2).flatten().sort() 
        # #a, _ = ((outputs - labels) ** 2).flatten().sort(descending=False) 
        # a, _ = ((outputs - labels) ** 2).flatten().sort(descending=True) 

        # #print(a) 

        # #sadfasdf

        # #print(len(((outputs - labels) ** 2).flatten())); print(len(a)) #a = a[0.2*len(((outputs - labels) ** 2).flatten())]  
        # #sadfs
        # #a = a[0.8*len(((outputs - labels) ** 2).flatten())] 
        # #a = a[round(0.2*len(((outputs - labels) ** 2).flatten()))]  
        # #a = a[round(0.2*len(((outputs - labels) ** 2).flatten()))]  
        # #a = a[round(0.8*len(((outputs - labels) ** 2).flatten()))]  
        # #a = a[round(0.8*len(((outputs - labels) ** 2).flatten()))]  
        # a = a[round(0.2*len(((outputs - labels) ** 2).flatten()))]  

        # #a = a[(0.8*len(((outputs - labels) ** 2).flatten())).floor()]  

        # #print(a)  

        # #asdfas

        # #a = a[round(0.8*len(((outputs - labels) ** 2).flatten()))]   
        # #print((torch.where((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)>a, torch.ones_like((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)), 0.*(((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape))).shape) 
        # #labels2new = torch.where((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)>a, torch.ones_like((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)), 0.*(((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape))
        # #labels2new = torch.where((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)>a, torch.zeros_like((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)), torch.ones_like((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)))
        # #labels2new = torch.where((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)>a, torch.zeros_like((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)), torch.ones_like((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)))
        # labels2new = torch.where((((outputs - labels) ** 2).flatten()).view(((outputs - labels) ** 2).shape)>a, torch.zeros_like((((outputs - labels) ** 2).flatten()).view(((outputs - labels) ** 2).shape)), torch.ones_like((((outputs - labels) ** 2).flatten()).view(((outputs - labels) ** 2).shape)))

        # #labels2new = torch.where((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)>=a, torch.zeros_like((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)), torch.ones_like((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)))
        # # # torch.where(condition, input, other)      

        # #print(((((outputs - labels) ** 2).flatten()).unflatten(dim=0, sizes=((outputs - labels) ** 2).shape)).shape) 

        # #print(labels2new)             
        # #sadfasdf 

        # #print(torch.count_nonzero(labels2new))    
        # #print(torch.count_nonzero(labels2new-1.))  
        # #sadfkzs



        # #print(torch.count_nonzero(labels2new))  
        # #print(torch.count_nonzero(labels2new-1.))  
        # #sadfkzs



        # # #arr1inds = arr1.argsort()
        # # #sorted_arr1 = arr1[arr1inds[::-1]]
        # # #sorted_arr2 = arr2[arr1inds[::-1]]

        # # arr1inds = ((outputs - labels) ** 2).flatten().argsort(descending=True) 
        # # #sorted_arr1 = arr1[arr1inds[::-1]] 
        # # #sorted_arr2 = arr2[arr1inds[::-1]]
        # # #sorted_arr = outputs[arr1inds]
        # # #sorted_arr = outputs[arr1inds]
        # # #sorted_arr = outputs[arr1inds]
        # # #sorted_arr = outputs2[arr1inds]
        # # #sorted_arr = outputs2.flatten()[arr1inds].view(outputs.shape)
        # # #sorted_arr = outputs.flatten()[arr1inds].view(outputs.shape)
        # # sorted_arr = outputs2.flatten()[arr1inds].view(outputs.shape)

        # # #print(outputs)                                          
        # # #print(labels2new)  

        # #loss += 0.1 * self.criterion(sorted_arr, labels2new)  
        # #loss += 0.1 * self.criterion(outputs, labels2new)
        # loss += 0.1 * self.criterion(outputs2, labels2new)

        # # # mse': 0.007522278829411348, 'mae': 0.04307622443723692, 'mave': 0.017571494465021993, 'acc': 0.9888189940164208, 'precision': 0.45436949719857384, 'recall': 0.027762249908265365, 'baseline_mse': 0.008060391471465861, 'f1': 0.052327271985781035  
        # # # mse': 0.01149324158238766, 'mae': 0.09229477474357345, 'mave': 0.08148465243129957, 'acc': 0.9893629486051028, 'precision': 0.608371201005457, 'recall': 0.12166443085105437, 'baseline_mse': 0.008060391471465861, 'f1': 0.20277677605481428
        
        
        
        

        # # # # loss = nn.L1Loss()                      
        # # # # input = torch.randn(3, 5, requires_grad=True) 
        # # # # target = torch.randn(3, 5) 
        # # # # output = loss(input, target)












        
        
        return loss
    


    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        
        #print(running_metric)                                                                    
        #sadfkzs 

        


        
        if (running_metric is not None) and (k is not None):
            metric_names = ['mse','mae','mave','acc','precision','recall','baseline_mse']
            # intermediary_values = ['mse','mae','mave','acc','tp','fp','fn','baseline_mse'] 

            final_metrics = {'mse':running_metric[0] / (k + 1), 'mae':running_metric[1] / (k + 1), 'mave':running_metric[2] / (k + 1), 'acc':running_metric[3]/ (k + 1), 'precision':running_metric[4]/(running_metric[4]+running_metric[5]), 'recall':running_metric[4]/(running_metric[4]+running_metric[6]), 'baseline_mse':running_metric[7] / (k + 1)}
            final_metrics['f1'] = 2 * final_metrics['precision'] * final_metrics['recall'] / (final_metrics['precision'] + final_metrics['recall'])

            return final_metrics

        elif (images == None) and (labels == None):
            intermediary_values = ['mse','mae','mave','acc','tp','fp','fn','baseline_mse']
            metric_init = np.zeros(len(intermediary_values)) # 
            return  metric_init
        
        
        else:
            
            #safsa





            #outputs = self.model(images)                                                       

            #outputs = self.model(images)  
            
            #outputs = self.model(images)
            
            outputs = self.model(images)
            #outputs, outputs2 = self.model(images)
            
            
            
            # print(images.shape)                                     
            # print(labels.shape)   
            # print(outputs.shape)
            # print(outputs2.shape)

            # asdfasdfzs



            
            #print(images.shape)                               
            #print(labels.shape)   
            #print(outputs.shape)
            
            #sadfasdf



            
            
            
            
            
            
            
            
            
            # # # # # # # # comment out till line 4369 # # # # # # # 4004                                                                                                                                                                                                                                             
            


            # formeanvartochange = []                                   

            # for vartochange in range(outputs.shape[0]): 

            #     # if vartochange == 0: 
            #     #     continue
            #     # if vartochange == 1:
            #     #     continue
            #     # if vartochange == 2:  
            #     #     continue
            #     # if vartochange == 3:    
            #     #     continue
            #     # if vartochange == 4:      
            #     #     continue
                
                
                
            #     # if vartochange == 0:                                           
            #     #     continue     
            #     # if vartochange == 1:
            #     #     continue
            #     # if vartochange == 2:  
            #     #     continue



                
                
            #     #if vartochange == 0:   
            #     #    continue







                
            #     #from metrics import compute_metrics_components                                                                   
            #     import sys         
            #     sys.path.append('../') 
            #     sys.path.append('../Automatic-Label-Error-Detection/')
            #     sys.path.append('../Automatic-Label-Error-Detection/src/')
            #     #from Automatic-Label-Error-Detection/src/metrics import compute_metrics_components            
            #     # from metrics import compute_metrics_components   
            #     # #print(outputss33.shape)
            #     # #print(labellss.shape)
                
            #     # # fromlabellss = 0.*np.ones_like(outputss33[vartochange,:,:,:].permute(1,2,0).detach().cpu().numpy())        
            #     # # for iuse in range(fromlabellss.shape[0]): 
            #     # #     for juse in range(fromlabellss.shape[1]):
            #     # #         fromlabellss[iuse, juse, labellss[vartochange, iuse, juse].detach().cpu().numpy()] = 1.
                
            #     # #print(fromlabellss)    
            #     # #print(fromlabellss.shape)
            #     # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # #_, components = compute_metrics_components(fromlabellss[:,:,:].copy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 

            #     # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # metriccss, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # #_, components = compute_metrics_components(fromlabellss[:,:,:].copy(), labellss[vartochange,:,:].clone().detach().cpu().numpy())  

            #     # #print(metriccss) 
            #     # #asdkfzsdf

                
                
            #     # # print('')  
            #     # # print(components)    
            #     # # print(components.shape) 
                
            #     # # print(components.max())                  
            #     # # print(-components.min())  
                
            #     # #asdfas


                
            #     # # from metrics import compute_IoU_component 
            #     # # theIoUs = compute_IoU_component(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy(), True)
            #     # # #theIoUs = compute_IoU_component(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), labellss[vartochange,:,:].clone().detach().cpu().numpy(), True)

            #     # # print(theIoUs)
            #     # # print(theIoUs.shape)

            #     # #asdfzsf

                
                
            #     # #import matplotlib.pyplot as plt                                    
            #     # plt.figure()                      
            #     # #plt.imshow(target_mask)             
            #     # #plt.imshow(infe)      
            #     # plt.imshow(components)  
            #     # #plt.imshow(components2) 
            #     # #plt.imshow(softmax_vs2)
            #     # plt.axis('off')     
            #     # plt.colorbar()
            #     # plt.savefig('/Data/temp/phileo17022024/phileo-bench/net_input2/img11.png', bbox_inches='tight')  
            #     #plt.savefig('./Image2_Components.png', bbox_inches='tight')                                    
            #     #plt.savefig('./Image2_ComponentsTrue.png', bbox_inches='tight')                                     
            #     #plt.savefig('./ImageResearch1c.png', bbox_inches='tight')           
            #     #plt.savefig('./ImageResearch1d.png', bbox_inches='tight')   
            #     #plt.savefig('./ImageResearch1e.png', bbox_inches='tight')      
            #     #plt.savefig('./ImageResearch1f.png', bbox_inches='tight') 



                
                
            #     # def mean_iou(labels, predictions, n_classes):
            #     #     mean_iou = 0.0
            #     #     seen_classes = 0

            #     #     for c in range(n_classes):
            #     #         labels_c = (labels == c)
            #     #         pred_c = (predictions == c)

            #     #         #print(labels_c)
            #     #         #print(labels_c.shape)
            #     #         #asdfasdf

            #     #         labels_c_sum = (labels_c).sum()
            #     #         pred_c_sum = (pred_c).sum()

            #     #         #print(labels_c_sum) 
            #     #         #sadfsadf

            #     #         if (labels_c_sum > 0) or (pred_c_sum > 0):
            #     #             seen_classes += 1 

            #     #             intersect = np.logical_and(labels_c, pred_c).sum()
            #     #             union = labels_c_sum + pred_c_sum - intersect

            #     #             print(intersect / union) 
            #     #             print(c)

            #     #             mean_iou += intersect / union 

            #     #     print(seen_classes)     

            #     #     return mean_iou / seen_classes if seen_classes else 0

            #     # #meaniou = mean_iou(labellss[vartochange,:,:].clone().detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy(), 11) 
            #     # meaniou = mean_iou(labellss[:,:,:].clone().detach().cpu().numpy(), theoutput[:,:,:].clone().detach().cpu().numpy(), 11)

            #     # #meaniou = mean_iou(labellss[:,:,:].clone().detach().cpu().numpy(), labellss[:,:,:].clone().detach().cpu().numpy(), 11) 
            #     # #meaniou = mean_iou(np.zeros_like(labellss[:,:,:].clone().detach().cpu().numpy()), labellss[:,:,:].clone().detach().cpu().numpy(), 11)
            #     # #meaniou = mean_iou(np.ones_like(labellss[:,:,:].clone().detach().cpu().numpy()), labellss[:,:,:].clone().detach().cpu().numpy(), 11)
            #     # #meaniou = mean_iou(theoutput[:,:,:].clone().detach().cpu().numpy(), theoutput[:,:,:].clone().detach().cpu().numpy(), 11)

            #     # print(meaniou)   
                
            #     #adfadsfs

                
                
                
                
                
                
            #     #adfas
            #     # # conftouse                                        
            #     #os.remove('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/Automatic-Label-Error-Detection/intermediate_results/components/*')         
            #     import glob    
            #     files = glob.glob('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/Automatic-Label-Error-Detection/intermediate_results/components/*')
            #     for f in files:
            #         os.remove(f)
            #     #os.remove('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/Automatic-Label-Error-Detection/intermediate_results/metrics/*') 
            #     files = glob.glob('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/Automatic-Label-Error-Detection/intermediate_results/metrics/*')
            #     for f in files:
            #         os.remove(f)

            #     #os.remove('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input/*')
            #     files = glob.glob('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input/*')
            #     for f in files:
            #         os.remove(f)
                
            #     # # (?)      
            #     # # (?) 
            #     # files = glob.glob('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/foldertodownload0/*')   
            #     # for f in files:
            #     #     os.remove(f)
            #     # # (?) 
            #     # # (?)

            #     #os.remove('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/logits/*')
            #     files = glob.glob('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/logits/*')
            #     for f in files:
            #         os.remove(f)
            #     #os.remove('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/inference_output/*') 
            #     files = glob.glob('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/inference_output/*') 
            #     for f in files:
            #         os.remove(f)
            #     #os.remove('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/gt_masks/*')
            #     files = glob.glob('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/gt_masks/*') 
            #     for f in files:
            #         os.remove(f)
            #     def clip_to_quantiles(arr, q_min=0.02, q_max=0.98):  
            #         return np.clip(arr,
            #             np.nanquantile(arr, q_min),
            #             np.nanquantile(arr, q_max),
            #         )      
            #     def render_s2_as_rgb(arr):  
            #         # If there are nodata values, lets cast them to zero.                      
            #         if np.ma.isMaskedArray(arr):         
            #             arr = np.ma.getdata(arr.filled(0)) 
            #         # # Select only Blue, green, and red.              
            #         #rgb_slice = arr[:, :, 0:3]  
            #         rgb_slice = arr        
            #         # Clip the data to the quantiles, so the RGB render is not stretched to outliers, 
            #         # Which produces dark images.    
            #         for c in [0, 1, 2]:
            #             rgb_slice[:, :, c] = clip_to_quantiles(rgb_slice[:, :, c])
            #         # The current slice is uint16, but we want an uint8 RGB render.       
            #         # We normalise the layer by dividing with the maximum value in the image.
            #         # Then we multiply it by 255 (the max of uint8) to be in the normal RGB range.
            #         for c in [0, 1, 2]: 
            #             rgb_slice[:, :, c] = (rgb_slice[:, :, c] / rgb_slice[:, :, c].max()) * 255.0
            #         # # We then round to the nearest integer and cast it to uint8.                       
            #         rgb_slice = np.rint(rgb_slice).astype(np.uint8)                
            #         return rgb_slice         
            #     fig = plt.figure()                                  
            #     #plt.imshow(render_s2_as_rgb(images[variabletemporary,:3,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]), interpolation="nearest")                
            #     plt.imshow(render_s2_as_rgb( images[vartochange,:3,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1] ), interpolation="nearest") 
            #     plt.axis('off')       
            #     #plt.savefig('net_input/img{fnam}.png'.format(fnam=1+variabletemporary+(i*images.shape[0])), bbox_inches='tight')    
            #     plt.savefig('net_input/img9.png', bbox_inches='tight')  
                
            #     #sadfasd



            #     # #np.save('logits/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())   
            #     # #np.save('logits/img9',  outputsmain[vartochange, :, :, :].detach().cpu().numpy() )  
            #     # #np.save('logits/img9',  outputsmain[vartochange, :, :, :].detach().cpu().numpy() )  
            #     # #np.save('inference_output/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputs[variabletemporary, :, :].detach().cpu().numpy()) 
            #     # np.save('inference_output/img9',  theoutput[vartochange, :, :].detach().cpu().numpy() ) 
            #     # #np.save('gt_masks/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())    
            #     # np.save('gt_masks/img9',  labellss[vartochange, :, :].detach().cpu().numpy().squeeze() )         
            #     # #np.save('gt_masks/img9',  theoutput[vartochange, :, :].detach().cpu().numpy().squeeze() )   
            #     # plt.close()                                                          
                
            #     # np.save('logits/img9',  outputsmain[vartochange, :, :, :].detach().cpu().numpy() )   

            #     # #adfzs
            #     # fig = plt.figure()                                     
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())          
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())  
            #     # plt.imshow(components)   
            #     # plt.axis('off')        
            #     # plt.savefig('net_input/img10.png', bbox_inches='tight') 
            #     # #azsdlf

            #     # componenttouse2 = np.ones_like(range(max(components.max(),-components.min())), dtype=object)   
            #     # #componenttouse2 = np.ones_like(range(components.max()), dtype=object) 
            #     # for i in range(len(componenttouse2)):
            #     #     componenttouse2[i] = [0.,]

            #     # for i in range(components.shape[0]):    
            #     #     for j in range(components.shape[1]): 
            #     #         if components[i, j] >= 0:  
            #     #             if componenttouse2[components[i, j]-1] == [0.,]:
            #     #                 componenttouse2[components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy(),]  
            #     #             else:
            #     #                 componenttouse2[components[i, j]-1].append(outputs3[vartochange, i, j].clone().detach().cpu().numpy()) 

            #     # componenttouse3 = []     
            #     # componenttouse4 = []
            #     # componenttouse5 = []
            #     # componenttouse6 = [] 
            #     # for componenttouse2a in componenttouse2:
            #     #     componenttouse3.append(np.nanmean(componenttouse2a))  
            #     #     componenttouse4.append(np.nanmedian(componenttouse2a))
            #     #     vartoadd = 0   
            #     #     for componenttouse2aa in componenttouse2a: 
            #     #         if componenttouse2aa >= 0.80:
            #     #             vartoadd += 1
            #     #     componenttouse5.append(vartoadd)
            #     #     componenttouse6.append(len(componenttouse2a))

            #     # for i in range(len(componenttouse5)):  
            #     #     componenttouse5[i] /= componenttouse6[i]   
            #     # del componenttouse6  

            #     # componentsmedian = 0.*np.ones_like(components)                  
            #     # for i in range(components.shape[0]):     
            #     #     for j in range(components.shape[1]): 
            #     #         if components[i, j] >= 0:  
            #     #             componentsmedian[i, j] = componenttouse4[components[i, j]-1]    
            #     #         else:
            #     #             componentsmedian[i, j] = -componenttouse4[-components[i, j]-1] 

            #     # components90at90 = 0.*np.ones_like(components)          
            #     # for i in range(components.shape[0]):     
            #     #     for j in range(components.shape[1]): 
            #     #         if components[i, j] >= 0:   
            #     #             if componenttouse5[components[i, j]-1] >= 0.80:
            #     #                 components90at90[i, j] = componenttouse5[components[i, j]-1]   
            #     #             else:
            #     #                 components90at90[i, j] = -componenttouse5[components[i, j]-1]
            #     #         else:
            #     #             components90at90[i, j] = -componenttouse5[-components[i, j]-1]

            #     # componentsmedian = 0.*np.ones_like(components)      
            #     # for i in range(components.shape[0]):       
            #     #     for j in range(components.shape[1]): 
            #     #         if components90at90[i, j] >= 0 and components[i, j] >= 0:  
            #     #             componentsmedian[i, j] = componenttouse4[components[i, j]-1]      
            #     #         elif components[i, j] >= 0:
            #     #             componentsmedian[i, j] = -componenttouse4[components[i, j]-1] 
            #     #         else:
            #     #             componentsmedian[i, j] = -componenttouse4[-components[i, j]-1]

            #     # fromlabellss = 0.*np.ones_like(outputss33[vartochange,:,:,:].permute(1,2,0).detach().cpu().numpy())     
            #     # for iuse in range(fromlabellss.shape[0]): 
            #     #     for juse in range(fromlabellss.shape[1]):
            #     #         #fromlabellss[iuse, juse, outputs[vartochange, iuse, juse].detach().cpu().numpy()] = componentsmedian[iuse, juse]    
            #     #         fromlabellss[iuse, juse, theoutput[vartochange, iuse, juse].detach().cpu().numpy()] = componentsmedian[iuse, juse]      
            #     # fromlabellss = np.transpose(fromlabellss, (2,0,1))

            #     # outputsmain2 = np.nanmean( np.array([ outputsmain[vartochange, :, :, :].detach().cpu().numpy(), fromlabellss ]), axis=0 )

            #     # #np.save('logits/img{fnam}'.format(fnam=1+variabletemporary+(imaini*images.shape[0])),  outputsmain2 )      
            #     # #np.save('logits/img9',  outputsmain[vartochange, :, :, :].detach().cpu().numpy() )  
            #     # np.save('logits/img9',  outputsmain2 )  
                
                






                
                
                

                
                
                
                
                
            #     # componenttouse2 = np.ones_like(range(max(components.max(),-components.min())), dtype=object)  
            #     # #componenttouse2 = np.ones_like(range(components.max()), dtype=object)
            #     # for i in range(len(componenttouse2)): 
            #     #     componenttouse2[i] = [0.,]

            #     # for i in range(components.shape[0]):      
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])         
            #     #         if components[i, j] >= 0:  
            #     #             #components[i, j] = conftouse[components[i, j]-1]                                
            #     #             #tempvariable = components[i, j]             
            #     #             #components[i, j] = conftouse[tempvariable-1] 
            #     #             #components[i, j] = conftouse[0]    
            #     #             #components2[i, j] = conftouse[0]
            #     #             #components2[i, j] = conftouse[components[i, j]-1]
            #     #             if componenttouse2[components[i, j]-1] == [0.,]:
            #     #                 #print(softmax_vs[i, j])
            #     #                 #componenttouse2[components[i, j]-1] = [softmax_vs[i, j],]  
            #     #                 #componenttouse2[components[i, j]-1] = [softmax_vs2[i, j],]  
            #     #                 componenttouse2[components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy(),]  
            #     #             else:
            #     #                 #componenttouse2[components[i, j]-1].append(softmax_vs[i, j])     
            #     #                 #componenttouse2[components[i, j]-1].append(softmax_vs2[i, j])  
            #     #                 #componenttouse2[components[i, j]-1].append(outputs3[0, i, j].clone().detach().cpu().numpy()) 
            #     #                 componenttouse2[components[i, j]-1].append(outputs3[vartochange, i, j].clone().detach().cpu().numpy()) 
            #     #             #componenttouse2[components[i, j]-1].append(softmax_vs[i, j]) 
            #     #             #print(conftouse[0]) 
            #     #             #print(components[i, j])              
            #     #             #print(components2[i, j])
            #     #         #else: 
            #     #         #    components2[i, j] = -conftouse[-components[i, j]-1] 

            #     # componenttouse3 = []   
            #     # componenttouse4 = []
            #     # componenttouse5 = []
            #     # componenttouse6 = [] 
                
            #     # for componenttouse2a in componenttouse2:
            #     #     #componenttouse3.append(np.mean(componenttouse2a))   
            #     #     componenttouse3.append(np.nanmean(componenttouse2a)) 
            #     #     #componenttouse4.append(np.median(componenttouse2a))
            #     #     componenttouse4.append(np.nanmedian(componenttouse2a))
            #     #     vartoadd = 0   
            #     #     for componenttouse2aa in componenttouse2a: 
            #     #         #if componenttouse2aa >= 0.9:         
            #     #         #if componenttouse2aa >= 0.8: 
            #     #         #if componenttouse2aa >= 0.85:
            #     #         if componenttouse2aa >= 0.80:
            #     #             vartoadd += 1
            #     #     componenttouse5.append(vartoadd)
            #     #     componenttouse6.append(len(componenttouse2a))

            #     # for i in range(len(componenttouse5)):  
            #     #     componenttouse5[i] /= componenttouse6[i]   
            #     # del componenttouse6  

            #     # #print(componenttouse5)   
            #     # #print(np.shape(componenttouse5)) 
            #     # #asdfzsdkf

            #     # #print(componenttouse4)
            #     # #print(np.shape(componenttouse4))
                


            #     # # componentsmean = 0.*np.ones_like(components)          

            #     # # for i in range(components.shape[0]):  
            #     # #     for j in range(components.shape[1]): 
            #     # #         #print(components[i, j])               
            #     # #         if components[i, j] >= 0:  
            #     # #             componentsmean[i, j] = componenttouse3[components[i, j]-1]   
            #     # #         else:
            #     # #             #print(components[i, j])
            #     # #             #print(-components[i, j]-1)
            #     # #             componentsmean[i, j] = -componenttouse3[-components[i, j]-1] 

            #     # # print(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())  
            #     # # print(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())
            #     # # print(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # # print(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # # #sadfsad

            #     # # #formeanvartochange.append(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())   
            #     # # #formeanvartochange.append(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())
            #     # # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # # # # Mean:        
            #     # # # # 0.90298937
            #     # # # # 0.90049725 

                
                
                
                
            #     # componentsmedian = 0.*np.ones_like(components)                  

            #     # for i in range(components.shape[0]):     
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])               
            #     #         if components[i, j] >= 0:  
            #     #             componentsmedian[i, j] = componenttouse4[components[i, j]-1]    
            #     #         else:
            #     #             componentsmedian[i, j] = -componenttouse4[-components[i, j]-1] 

            #     # # print(componentsmedian[infe==gt_masks][components[infe==gt_masks]>=0.].mean())            
            #     # # print(componentsmedian[infe!=gt_masks][components[infe!=gt_masks]>=0.].mean())  
            #     # # print(np.median(componentsmedian[infe==gt_masks][components[infe==gt_masks]>=0.]))   
            #     # # print(np.median(componentsmedian[infe!=gt_masks][components[infe!=gt_masks]>=0.]))

            #     # #print(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean()) 
            #     # #print(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())
            #     # #print(np.median(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # #print(np.median(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # #adfasdf

            #     # #formeanvartochange.append(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())    
            #     # #formeanvartochange.append(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean()) 
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # # # Median:              
            #     # # # 0.95357752  
            #     # # # 0.94796602

                
                
                
                
            #     # components90at90 = 0.*np.ones_like(components)          

            #     # for i in range(components.shape[0]):     
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])                         
            #     #         if components[i, j] >= 0:   
            #     #             #if componenttouse5[components[i, j]-1] >= 0.9:           
            #     #             #if componenttouse5[components[i, j]-1] >= 0.8:  
            #     #             #if componenttouse5[components[i, j]-1] >= 0.85: 
            #     #             if componenttouse5[components[i, j]-1] >= 0.80:
            #     #                 components90at90[i, j] = componenttouse5[components[i, j]-1]   
            #     #             else:
            #     #                 components90at90[i, j] = -componenttouse5[components[i, j]-1]
            #     #         else:
            #     #             components90at90[i, j] = -componenttouse5[-components[i, j]-1]

            #     # componentsmedian = 0.*np.ones_like(components)      

            #     # for i in range(components.shape[0]):       
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])                                                                  
            #     #         #if components[i, j] >= 0:
            #     #         if components90at90[i, j] >= 0 and components[i, j] >= 0:  
            #     #             componentsmedian[i, j] = componenttouse4[components[i, j]-1]      
            #     #         elif components[i, j] >= 0:
            #     #             componentsmedian[i, j] = -componenttouse4[components[i, j]-1] 
            #     #         else:
            #     #             componentsmedian[i, j] = -componenttouse4[-components[i, j]-1]
            #     # #sadfasd

            #     # #print(componentsmedian)      
            #     # #print(componentsmedian.shape) 
            #     # #sadfasdf

            #     # fromlabellss = 0.*np.ones_like(outputss33[vartochange,:,:,:].permute(1,2,0).detach().cpu().numpy())     
            #     # for iuse in range(fromlabellss.shape[0]): 
            #     #     for juse in range(fromlabellss.shape[1]):
            #     #         #fromlabellss[iuse, juse, labellss[vartochange, iuse, juse].detach().cpu().numpy()] = 1.     
            #     #         #fromlabellss[iuse, juse, labellss[vartochange, iuse, juse].detach().cpu().numpy()] = componentsmedian[iuse, juse]     
            #     #         fromlabellss[iuse, juse, outputs[vartochange, iuse, juse].detach().cpu().numpy()] = componentsmedian[iuse, juse]

            #     # fromlabellss = np.transpose(fromlabellss, (2,0,1)) 

            #     # #print(fromlabellss)   
            #     # #print(fromlabellss.shape)

            #     # #print(outputsmain[vartochange, :, :, :].detach().cpu().numpy().shape) 
            #     # #print(fromlabellss.shape)
            #     # #asdfas

            #     # outputsmain2 = np.nanmean( np.array([ outputsmain[vartochange, :, :, :].detach().cpu().numpy(), fromlabellss ]), axis=0 )

            #     # #print(outputsmain2)
            #     # #print(outputsmain2.shape) 

            #     # #plt.savefig('../net_input/img{fnam}.png'.format(fnam=1+variabletemporary+(i*images.shape[0])), bbox_inches='tight')     

            #     # #np.save('../logits/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())  
            #     # #np.save('../logits/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputsmain2)

            #     # #np.save('logits/img9',  outputsmain[vartochange, :, :, :].detach().cpu().numpy() )     
            #     # np.save('logits/img9',  outputsmain2 )

























                
                
                
                
                
                
                
                
                
                
                
            #     # #print(outputsmain[vartochange, :, :, :].detach().cpu().numpy())   
            #     # #print(outputsmain[vartochange, :, :, :].detach().cpu().numpy().shape)
                
            #     # #asdfas
            #     # componenttouse2 = np.ones_like(range(max(components.max(),-components.min())), dtype=object) 
            #     # #componenttouse2 = np.ones_like(range(components.max()), dtype=object)
            #     # for i in range(len(componenttouse2)):
            #     #     componenttouse2[i] = [0.,]

            #     # for i in range(components.shape[0]):  
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])         
            #     #         if components[i, j] >= 0:  
            #     #             #components[i, j] = conftouse[components[i, j]-1]                                   
            #     #             #tempvariable = components[i, j]             
            #     #             #components[i, j] = conftouse[tempvariable-1] 
            #     #             #components[i, j] = conftouse[0]    
            #     #             #components2[i, j] = conftouse[0]
            #     #             #components2[i, j] = conftouse[components[i, j]-1]
            #     #             if componenttouse2[components[i, j]-1] == [0.,]:
            #     #                 #print(softmax_vs[i, j])
            #     #                 #componenttouse2[components[i, j]-1] = [softmax_vs[i, j],]  
            #     #                 #componenttouse2[components[i, j]-1] = [softmax_vs2[i, j],]  
            #     #                 componenttouse2[components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy(),]  
            #     #             else:
            #     #                 #componenttouse2[components[i, j]-1].append(softmax_vs[i, j])     
            #     #                 #componenttouse2[components[i, j]-1].append(softmax_vs2[i, j])  
            #     #                 #componenttouse2[components[i, j]-1].append(outputs3[0, i, j].clone().detach().cpu().numpy()) 
            #     #                 componenttouse2[components[i, j]-1].append(outputs3[vartochange, i, j].clone().detach().cpu().numpy()) 
            #     #             #componenttouse2[components[i, j]-1].append(softmax_vs[i, j]) 
            #     #             #print(conftouse[0]) 
            #     #             #print(components[i, j])              
            #     #             #print(components2[i, j])
            #     #         #else: 
            #     #         #    components2[i, j] = -conftouse[-components[i, j]-1] 

            #     # componenttouse3 = []   
            #     # componenttouse4 = []
            #     # componenttouse5 = []
            #     # componenttouse6 = [] 
                
            #     # for componenttouse2a in componenttouse2:
            #     #     #componenttouse3.append(np.mean(componenttouse2a))   
            #     #     componenttouse3.append(np.nanmean(componenttouse2a)) 
            #     #     #componenttouse4.append(np.median(componenttouse2a))
            #     #     componenttouse4.append(np.nanmedian(componenttouse2a))
            #     #     vartoadd = 0   
            #     #     for componenttouse2aa in componenttouse2a: 
            #     #         #if componenttouse2aa >= 0.9:         
            #     #         #if componenttouse2aa >= 0.8: 
            #     #         #if componenttouse2aa >= 0.85:
            #     #         if componenttouse2aa >= 0.80:
            #     #             vartoadd += 1
            #     #     componenttouse5.append(vartoadd)
            #     #     componenttouse6.append(len(componenttouse2a))

            #     # for i in range(len(componenttouse5)):  
            #     #     componenttouse5[i] /= componenttouse6[i]   
            #     # del componenttouse6  

            #     # #print(componenttouse5)   
            #     # #print(np.shape(componenttouse5)) 
            #     # #asdfzsdkf

            #     # #print(componenttouse4)
            #     # #print(np.shape(componenttouse4))
                


            #     # # componentsmean = 0.*np.ones_like(components)          

            #     # # for i in range(components.shape[0]):  
            #     # #     for j in range(components.shape[1]): 
            #     # #         #print(components[i, j])               
            #     # #         if components[i, j] >= 0:  
            #     # #             componentsmean[i, j] = componenttouse3[components[i, j]-1]   
            #     # #         else:
            #     # #             #print(components[i, j])
            #     # #             #print(-components[i, j]-1)
            #     # #             componentsmean[i, j] = -componenttouse3[-components[i, j]-1] 

            #     # # print(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())  
            #     # # print(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())
            #     # # print(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # # print(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # # #sadfsad

            #     # # #formeanvartochange.append(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())   
            #     # # #formeanvartochange.append(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())
            #     # # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # # # # Mean:        
            #     # # # # 0.90298937
            #     # # # # 0.90049725 

                
                
                
                
            #     # componentsmedian = 0.*np.ones_like(components)                  

            #     # for i in range(components.shape[0]):     
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])               
            #     #         if components[i, j] >= 0:  
            #     #             componentsmedian[i, j] = componenttouse4[components[i, j]-1]    
            #     #         else:
            #     #             componentsmedian[i, j] = -componenttouse4[-components[i, j]-1] 

            #     # # print(componentsmedian[infe==gt_masks][components[infe==gt_masks]>=0.].mean())            
            #     # # print(componentsmedian[infe!=gt_masks][components[infe!=gt_masks]>=0.].mean())  
            #     # # print(np.median(componentsmedian[infe==gt_masks][components[infe==gt_masks]>=0.]))   
            #     # # print(np.median(componentsmedian[infe!=gt_masks][components[infe!=gt_masks]>=0.]))

            #     # #print(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean()) 
            #     # #print(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())
            #     # #print(np.median(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # #print(np.median(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # #adfasdf

            #     # #formeanvartochange.append(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())    
            #     # #formeanvartochange.append(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean()) 
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # # # Median:              
            #     # # # 0.95357752  
            #     # # # 0.94796602

                
                
                
                
            #     # components90at90 = 0.*np.ones_like(components)          

            #     # for i in range(components.shape[0]):     
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])                         
            #     #         if components[i, j] >= 0:   
            #     #             #if componenttouse5[components[i, j]-1] >= 0.9:           
            #     #             #if componenttouse5[components[i, j]-1] >= 0.8:  
            #     #             #if componenttouse5[components[i, j]-1] >= 0.85: 
            #     #             if componenttouse5[components[i, j]-1] >= 0.80:
            #     #                 components90at90[i, j] = componenttouse5[components[i, j]-1]   
            #     #             else:
            #     #                 components90at90[i, j] = -componenttouse5[components[i, j]-1]
            #     #         else:
            #     #             components90at90[i, j] = -componenttouse5[-components[i, j]-1]

            #     # componentsmedian = 0.*np.ones_like(components)      

            #     # for i in range(components.shape[0]):       
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])                                                                  
            #     #         #if components[i, j] >= 0:
            #     #         if components90at90[i, j] >= 0 and components[i, j] >= 0:  
            #     #             componentsmedian[i, j] = componenttouse4[components[i, j]-1]      
            #     #         elif components[i, j] >= 0:
            #     #             componentsmedian[i, j] = -componenttouse4[components[i, j]-1] 
            #     #         else:
            #     #             componentsmedian[i, j] = -componenttouse4[-components[i, j]-1]
            #     # #sadfasd

            #     # #print(componentsmedian)      
            #     # #print(componentsmedian.shape) 
            #     # #sadfasdf

            #     # fromlabellss = 0.*np.ones_like(outputss33[vartochange,:,:,:].permute(1,2,0).detach().cpu().numpy())     
            #     # for iuse in range(fromlabellss.shape[0]): 
            #     #     for juse in range(fromlabellss.shape[1]):
            #     #         #fromlabellss[iuse, juse, labellss[vartochange, iuse, juse].detach().cpu().numpy()] = 1.     
            #     #         #fromlabellss[iuse, juse, labellss[vartochange, iuse, juse].detach().cpu().numpy()] = componentsmedian[iuse, juse]     
            #     #         fromlabellss[iuse, juse, outputs[vartochange, iuse, juse].detach().cpu().numpy()] = componentsmedian[iuse, juse]

            #     # fromlabellss = np.transpose(fromlabellss, (2,0,1)) 

            #     # #print(fromlabellss)   
            #     # #print(fromlabellss.shape)

            #     # #print(outputsmain[vartochange, :, :, :].detach().cpu().numpy().shape) 
            #     # #print(fromlabellss.shape)
            #     # #asdfas

            #     # outputsmain2 = np.nanmean( np.array([ outputsmain[vartochange, :, :, :].detach().cpu().numpy(), fromlabellss ]), axis=0 )

            #     # #print(outputsmain2)
            #     # #print(outputsmain2.shape) 

            #     # #plt.savefig('../net_input/img{fnam}.png'.format(fnam=1+variabletemporary+(i*images.shape[0])), bbox_inches='tight')     

            #     # #np.save('../logits/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())  
            #     # #np.save('../logits/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputsmain2)

            #     # #np.save('logits/img9',  outputsmain[vartochange, :, :, :].detach().cpu().numpy() )     
            #     # np.save('logits/img9',  outputsmain2 )  
                
            #     # #np.save('../inference_output/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputs[variabletemporary, :, :].detach().cpu().numpy()) 
                
            #     # #np.save('../gt_masks/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())


                
                
                
            #     # #print(outputsmain[vartochange, :, :, :].detach().cpu().numpy()) 
            #     # #print(outputsmain[vartochange, :, :, :].detach().cpu().numpy().shape)
                
            #     # #asdfas
            #     # componenttouse2 = np.ones_like(range(max(components.max(),-components.min())), dtype=object) 
            #     # #componenttouse2 = np.ones_like(range(components.max()), dtype=object)
            #     # for i in range(len(componenttouse2)):
            #     #     componenttouse2[i] = [0.,]

            #     # for i in range(components.shape[0]):  
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])         
            #     #         if components[i, j] >= 0:  
            #     #             #components[i, j] = conftouse[components[i, j]-1]                              
            #     #             #tempvariable = components[i, j]             
            #     #             #components[i, j] = conftouse[tempvariable-1] 
            #     #             #components[i, j] = conftouse[0]    
            #     #             #components2[i, j] = conftouse[0]
            #     #             #components2[i, j] = conftouse[components[i, j]-1]
            #     #             if componenttouse2[components[i, j]-1] == [0.,]:
            #     #                 #print(softmax_vs[i, j])
            #     #                 #componenttouse2[components[i, j]-1] = [softmax_vs[i, j],]  
            #     #                 #componenttouse2[components[i, j]-1] = [softmax_vs2[i, j],]  
            #     #                 componenttouse2[components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy(),]  
            #     #             else:
            #     #                 #componenttouse2[components[i, j]-1].append(softmax_vs[i, j])     
            #     #                 #componenttouse2[components[i, j]-1].append(softmax_vs2[i, j])  
            #     #                 #componenttouse2[components[i, j]-1].append(outputs3[vartochange, i, j].clone().detach().cpu().numpy()) 
            #     #                 componenttouse2[components[i, j]-1].append(outputs3[vartochange, i, j].clone().detach().cpu().numpy()) 
            #     #             #componenttouse2[components[i, j]-1].append(softmax_vs[i, j]) 
            #     #             #print(conftouse[0]) 
            #     #             #print(components[i, j])             
            #     #             #print(components2[i, j])
            #     #         #else: 
            #     #         #    components2[i, j] = -conftouse[-components[i, j]-1] 

            #     # componenttouse3 = []   
            #     # componenttouse4 = []
            #     # componenttouse5 = []
            #     # componenttouse6 = [] 
                
            #     # for componenttouse2a in componenttouse2:
            #     #     #componenttouse3.append(np.mean(componenttouse2a))   
            #     #     componenttouse3.append(np.nanmean(componenttouse2a)) 
            #     #     #componenttouse4.append(np.median(componenttouse2a))
            #     #     componenttouse4.append(np.nanmedian(componenttouse2a))
            #     #     vartoadd = 0   
            #     #     for componenttouse2aa in componenttouse2a: 
            #     #         #if componenttouse2aa >= 0.9:         
            #     #         #if componenttouse2aa >= 0.8: 
            #     #         #if componenttouse2aa >= 0.85:
            #     #         if componenttouse2aa >= 0.80:
            #     #             vartoadd += 1
            #     #     componenttouse5.append(vartoadd)
            #     #     componenttouse6.append(len(componenttouse2a))

            #     # for i in range(len(componenttouse5)):  
            #     #     componenttouse5[i] /= componenttouse6[i]   
            #     # del componenttouse6  

            #     # #print(componenttouse5)   
            #     # #print(np.shape(componenttouse5)) 
            #     # #asdfzsdkf

            #     # #print(componenttouse4)
            #     # #print(np.shape(componenttouse4))
                


            #     # # componentsmean = 0.*np.ones_like(components)          

            #     # # for i in range(components.shape[0]):  
            #     # #     for j in range(components.shape[1]): 
            #     # #         #print(components[i, j])               
            #     # #         if components[i, j] >= 0:  
            #     # #             componentsmean[i, j] = componenttouse3[components[i, j]-1]   
            #     # #         else:
            #     # #             #print(components[i, j])
            #     # #             #print(-components[i, j]-1)
            #     # #             componentsmean[i, j] = -componenttouse3[-components[i, j]-1] 

            #     # # print(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())  
            #     # # print(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())
            #     # # print(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # # print(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # # #sadfsad

            #     # # #formeanvartochange.append(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())   
            #     # # #formeanvartochange.append(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())
            #     # # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # # # # Mean:        
            #     # # # # 0.90298937
            #     # # # # 0.90049725 

                
                
                
                
            #     # componentsmedian = 0.*np.ones_like(components)                   

            #     # for i in range(components.shape[0]):     
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])               
            #     #         if components[i, j] >= 0:  
            #     #             componentsmedian[i, j] = componenttouse4[components[i, j]-1]    
            #     #         else:
            #     #             componentsmedian[i, j] = -componenttouse4[-components[i, j]-1] 

            #     # # print(componentsmedian[infe==gt_masks][components[infe==gt_masks]>=0.].mean())            
            #     # # print(componentsmedian[infe!=gt_masks][components[infe!=gt_masks]>=0.].mean())  
            #     # # print(np.median(componentsmedian[infe==gt_masks][components[infe==gt_masks]>=0.]))   
            #     # # print(np.median(componentsmedian[infe!=gt_masks][components[infe!=gt_masks]>=0.]))

            #     # #print(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean()) 
            #     # #print(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())
            #     # #print(np.median(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # #print(np.median(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # #adfasdf

            #     # #formeanvartochange.append(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())    
            #     # #formeanvartochange.append(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean()) 
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # # # Median:              
            #     # # # 0.95357752  
            #     # # # 0.94796602

                
                
                
                
            #     # components90at90 = 0.*np.ones_like(components)           

            #     # for i in range(components.shape[0]):     
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])                         
            #     #         if components[i, j] >= 0:   
            #     #             #if componenttouse5[components[i, j]-1] >= 0.9:           
            #     #             #if componenttouse5[components[i, j]-1] >= 0.8:  
            #     #             #if componenttouse5[components[i, j]-1] >= 0.85: 
            #     #             if componenttouse5[components[i, j]-1] >= 0.80:
            #     #                 components90at90[i, j] = componenttouse5[components[i, j]-1]   
            #     #             else:
            #     #                 components90at90[i, j] = -componenttouse5[components[i, j]-1]
            #     #         else:
            #     #             components90at90[i, j] = -componenttouse5[-components[i, j]-1]

            #     # componentsmedian = 0.*np.ones_like(components)      

            #     # for i in range(components.shape[0]):       
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])                                                                 
            #     #         #if components[i, j] >= 0:
            #     #         if components90at90[i, j] >= 0 and components[i, j] >= 0:  
            #     #             componentsmedian[i, j] = componenttouse4[components[i, j]-1]      
            #     #         elif components[i, j] >= 0:
            #     #             componentsmedian[i, j] = -componenttouse4[components[i, j]-1] 
            #     #         else:
            #     #             componentsmedian[i, j] = -componenttouse4[-components[i, j]-1]
            #     # #sadfasd

            #     # #print(componentsmedian)      
            #     # #print(componentsmedian.shape) 
            #     # #sadfasdf

            #     # fromlabellss = 0.*np.ones_like(outputss33[vartochange,:,:,:].permute(1,2,0).detach().cpu().numpy())     
            #     # for iuse in range(fromlabellss.shape[0]): 
            #     #     for juse in range(fromlabellss.shape[1]):
            #     #         #fromlabellss[iuse, juse, labellss[vartochange, iuse, juse].detach().cpu().numpy()] = 1.     
            #     #         fromlabellss[iuse, juse, labellss[vartochange, iuse, juse].detach().cpu().numpy()] = componentsmedian[iuse, juse]     

            #     # fromlabellss = np.transpose(fromlabellss, (2,0,1))

            #     # #print(fromlabellss)   
            #     # #print(fromlabellss.shape)

            #     # #print(outputsmain[vartochange, :, :, :].detach().cpu().numpy().shape) 
            #     # #print(fromlabellss.shape)
            #     # #asdfas

            #     # outputsmain2 = np.nanmean( np.array([ outputsmain[vartochange, :, :, :].detach().cpu().numpy(), fromlabellss ]), axis=0 )

            #     # #print(outputsmain2)
            #     # #print(outputsmain2.shape) 

            #     # #plt.savefig('../net_input/img{fnam}.png'.format(fnam=1+variabletemporary+(i*images.shape[0])), bbox_inches='tight')     

            #     # #np.save('../logits/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())  
            #     # np.save('../logits/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputsmain2)

            #     # #np.save('../inference_output/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputs[variabletemporary, :, :].detach().cpu().numpy()) 
                
            #     # #np.save('../gt_masks/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())
                
                
                
                
                
            #     # fromlabellss = 0.*np.ones_like(outputss33[vartochange,:,:,:].permute(1,2,0).detach().cpu().numpy())    
            #     # for iuse in range(fromlabellss.shape[0]): 
            #     #     for juse in range(fromlabellss.shape[1]):
            #     #         fromlabellss[iuse, juse, labellss[vartochange, iuse, juse].detach().cpu().numpy()] = 1.
                
            #     # fromlabellss = np.transpose(fromlabellss, (2,0,1))

            #     # #print(fromlabellss)    
            #     # print(fromlabellss.shape)

            #     # asdfas
            #     # plt.savefig('../net_input/img{fnam}.png'.format(fnam=1+variabletemporary+(i*images.shape[0])), bbox_inches='tight')     

            #     # np.save('../logits/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())  
                
            #     # np.save('../inference_output/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputs[variabletemporary, :, :].detach().cpu().numpy()) 
                
            #     # np.save('../gt_masks/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())
            #     # asdf



                
                
                
                
            #     # # (?)                        
            #     # if vartochange == 0:     
            #     #     continue
            #     # elif vartochange == 1:    
            #     #    continue
            #     # # (?)

            #     from analyze_metrics import evaluate                         
            #     # conftouse = evaluate()

            #     # print(conftouse)         
            #     # print(np.shape(conftouse))

            #     # components2 = 0.*np.ones_like(components)                                          

            #     # for i in range(components.shape[0]):       
            #     #     for j in range(components.shape[1]):
            #     #         #print(components[i, j])            
            #     #         #sadfsadfas
            #     #         if components[i, j] >= 0:       
            #     #             #components[i, j] = conftouse[components[i, j]-1]                                                                                                                      
            #     #             #tempvariable = components[i, j]                     
            #     #             #components[i, j] = conftouse[tempvariable-1]  
            #     #             #components[i, j] = conftouse[0]         
            #     #             #components2[i, j] = conftouse[0]
            #     #             components2[i, j] = conftouse[components[i, j]-1]             
            #     #             #print(conftouse[0])    
            #     #             #print(components[i, j])                                         
            #     #             #print(components2[i, j])  
            #     #             #sadfaszdf
            #     #         else:
            #     #             #components2[i, j] = -conftouse[-components[i, j]-1]
            #     #             components2[i, j] = conftouse[-components[i, j]-1]   

            #     # #print(components)                                                                         
            #     # #print(components.shape)   
                
            #     # #print(components2)     
            #     # #print(components2.shape)
                
            #     # # #adfzs
            #     # fig = plt.figure()                                     
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                                  
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())   
            #     # #plt.imshow(components)   
            #     # plt.imshow(components2)   
            #     # plt.axis('off')        
            #     # plt.colorbar() 
            #     # #plt.savefig('net_input/img10.png', bbox_inches='tight')    
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10b.png', bbox_inches='tight')                      
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10c.png', bbox_inches='tight')      
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10d.png', bbox_inches='tight')  
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10e.png', bbox_inches='tight')  
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10f.png', bbox_inches='tight')  
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10g.png', bbox_inches='tight')  
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10h.png', bbox_inches='tight')  
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10i.png', bbox_inches='tight')  
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10k.png', bbox_inches='tight')  
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10l.png', bbox_inches='tight')  
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10m.png', bbox_inches='tight')  
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10n.png', bbox_inches='tight')  
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_1.png', bbox_inches='tight')   
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_3.png', bbox_inches='tight')  
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_5.png', bbox_inches='tight')  
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_7.png', bbox_inches='tight')  
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9.png', bbox_inches='tight')  
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bb.png', bbox_inches='tight')  
            #     # plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bbbb.png', bbox_inches='tight')  
            #     # #plt.savefig('net_input2/img10c.png', bbox_inches='tight')  
            #     # # #azsdlf

            #     # fig = plt.figure()                                               
            #     # #plt.imshow(np.where(components2 < 0.10, components2, float("nan")))                                                                  
            #     # #plt.imshow(np.where(components2 < 0.20, components2, float("nan")))                                      
            #     # #plt.imshow(np.where(components2 > 1.10, components2, float("nan")))                                  
            #     # #plt.imshow(np.where(components2 < 0.40, components2, float("nan")))                                  
            #     # #plt.imshow(np.where(components2 < 0.41, components2, float("nan")))                                  
            #     # #plt.imshow(np.where(components2 < 0.30, components2, float("nan")))                                  
            #     # #plt.imshow(np.where(components2 < 0.31, components2, float("nan")))                                  
            #     # plt.imshow(np.where(components2 < 0.20, components2, float("nan")))                                  
            #     # #plt.imshow(np.where(components2 < 0.30, components2, float("nan")))                                  
            #     # plt.axis('off')                      
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_2.png', bbox_inches='tight')                     
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_4.png', bbox_inches='tight')    
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_6.png', bbox_inches='tight')
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_8.png', bbox_inches='tight')
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9b.png', bbox_inches='tight')
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bbb.png', bbox_inches='tight')
            #     # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bbbbb.png', bbox_inches='tight')
            #     # plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bbbbbb.png', bbox_inches='tight')
                
            #     # # fig = plt.figure()                                             
            #     # # plt.imshow(np.where(confmetric > 0.2, confmetric, float("nan")))                              
            #     # # plt.axis('off')                  
            #     # # plt.savefig('foldertodownload%s/confmetricgre0.2b.png'%str(vartochange), bbox_inches='tight') 

            #     # plt.close('all') 

            #     # asdfas



            #     #conftouse = evaluate()         
            #     # try:         
            #     #     conftouse = evaluate()             
            #     # except: 
            #     #     continue 

            #     #asdfzsdkf
                
            #     #conftouse = [4.4375, 2.466666666666667, 10.358288770053475, 8.101321585903083, 43.61994459833795, 2.642857142857143, 0.6666666666666666, 7.84, 0.0, 1.25, 18.263157894736842, 73.87249498997996, 16.0, 26.445288753799392, 86.37382661512976, 38.5, 41.1063829787234, 42.0, 53.13592233009709, 54.44444444444444, 66.8273381294964, 66.125, 97.86363636363636, 100.33333333333333, 116.05242718446603, 104.70588235294117, 118.09152542372881, 108.0, 117.01785714285714, 121.41545893719807, 124.11111111111111, 124.85416666666667, 122.83333333333333]

            #     #conftouse = [4.5625, 9.966666666666667, 8.67379679144385, 25.502202643171806, 71.50544783010157, 54.142857142857146, 59.833333333333336, 76.82, 89.5, 125.1875, 22.92105263157895, 24.0936873747495, 4.0, 69.64589665653496, 99.07979017117614, 19.0, 69.06382978723404, 119.0, 63.213592233009706, 124.57777777777778, 118.22302158273381, 97.875, 4.136363636363637, 16.0, 43.56504854368932, 67.29411764705883, 115.76271186440678, 64.0, 104.875, 7.256038647342995, 24.525252525252526, 42.604166666666664, 126.0]

            #     #print(metriccss)   
            #     #print(metriccss['mean_x'])       
            #     #print(metriccss['mean_y']) 
            #     #asdfasdf

                
                
            #     # from metrics import compute_metrics_components   
            #     # #print(outputss33.shape)
            #     # #print(labellss.shape)
                
            #     # # fromlabellss = 0.*np.ones_like(outputss33[vartochange,:,:,:].permute(1,2,0).detach().cpu().numpy())        
            #     # # for iuse in range(fromlabellss.shape[0]): 
            #     # #     for juse in range(fromlabellss.shape[1]):
            #     # #         fromlabellss[iuse, juse, labellss[vartochange, iuse, juse].detach().cpu().numpy()] = 1.
                
            #     # #print(fromlabellss)    
            #     # #print(fromlabellss.shape)
            #     # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # #_, components = compute_metrics_components(fromlabellss[:,:,:].copy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 

            #     # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # metriccss, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # #_, components = compute_metrics_components(fromlabellss[:,:,:].copy(), labellss[vartochange,:,:].clone().detach().cpu().numpy())  

            #     # #print(metriccss)
            #     # #print(metriccss.keys())  
            #     # #asdkfzsdf

            #     # #print(metriccss.keys())  
            #     # #asdfzsdf

            #     # # # D_rel 
            #     # # # D_var
            #     # # # D_var_in
            #     # # # (?)
                
                
                
                
            #     # #metriccss['E_in']                       
            #     # #metriccss['D_in']    
            #     # #metriccss['V_in']

            #     # # # dict_keys(['iou', 'iou0', 'class', 'mean_x', 'mean_y', 'E', 'E_in', 'E_bd', 
            #     # # # 'E_rel', 'E_rel_in', 'E_var', 'E_var_in', 'E_var_bd', 'E_var_rel', 'E_var_rel_in', 
            #     # # # 'D', 'D_in', 'D_bd', 'D_rel', 'D_rel_in', 'D_var', 'D_var_in', 'D_var_bd', 
            #     # # # 'D_var_rel', 'D_var_rel_in', 'V', 'V_in', 'V_bd', 'V_rel', 'V_rel_in', 'V_var', 
            #     # # # 'V_var_in', 'V_var_bd', 'V_var_rel', 'V_var_rel_in', 'S', 'S_in', 'S_bd', 'S_rel', 
            #     # # # 'S_rel_in', 'cprob0', 'cprob1', 'cprob2', 'cprob3', 'cprob4', 'cprob5', 'cprob6', 
            #     # # # 'cprob7', 'cprob8', 'cprob9', 'cprob10', 'ndist0', 'ndist1', 'ndist2', 'ndist3', 
            #     # # # 'ndist4', 'ndist5', 'ndist6', 'ndist7', 'ndist8', 'ndist9', 'ndist10'])  

            #     # #sdfsa



            #     # # print('')    
            #     # # print(components)    
            #     # # print(components.shape) 
                
            #     # # print(components.max())                  
            #     # # print(-components.min())  
                
            #     # #asdfas


                
            #     # # from metrics import compute_IoU_component 
            #     # # theIoUs = compute_IoU_component(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy(), True)
            #     # # #theIoUs = compute_IoU_component(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), labellss[vartochange,:,:].clone().detach().cpu().numpy(), True)

            #     # # print(theIoUs)
            #     # # print(theIoUs.shape)

            #     # #asdfzsf

                
                
            #     # # #import matplotlib.pyplot as plt                                     
            #     # # plt.figure()                      
            #     # # #plt.imshow(target_mask)             
            #     # # #plt.imshow(infe)      
            #     # # plt.imshow(components)  
            #     # # #plt.imshow(components2) 
            #     # # #plt.imshow(softmax_vs2)
            #     # # plt.axis('off')     
            #     # # plt.colorbar()
            #     # # plt.savefig('/Data/temp/phileo17022024/phileo-bench/net_input2/img11.png', bbox_inches='tight')

                
                
            #     # # # (?)                                          
            #     # # if vartochange == 0:       
            #     # #     continue
            #     # # elif vartochange == 1:     
            #     # #    continue 
            #     # # # (?)
                
            #     # #conftouse = np.array(metriccss['mean_y'])       
            #     # #conftouse = np.array(metriccss['D'])   
                
                
                
            #     # conftouse = np.array(metriccss['D_in']) 
            #     # conftouse = (conftouse-np.min(conftouse))/(np.max(conftouse)-np.min(conftouse))
            #     # conftouse = 1. - conftouse

            #     # conftoussee = np.array(metriccss['E_in']) 
            #     # conftoussee = (conftoussee-np.min(conftoussee))/(np.max(conftoussee)-np.min(conftoussee))
            #     # conftoussee = 1. - conftoussee
            #     # conftouse += conftoussee 
                
            #     # conftoussee = np.array(metriccss['V_in'])  
            #     # conftoussee = (conftoussee-np.min(conftoussee))/(np.max(conftoussee)-np.min(conftoussee))
            #     # conftoussee = 1. - conftoussee
            #     # conftouse += conftoussee
                
            #     # conftouse /= 3     

                
                
            #     # # conftouse = conftouse = np.array(metriccss['D_var']) 
            #     # # conftouse = (conftouse-np.min(conftouse))/(np.max(conftouse)-np.min(conftouse))

            #     # # #metriccss['E_in']                                                            
            #     # # #metriccss['D_in']      
            #     # # #metriccss['V_in'] 

            #     # # # # dict_keys(['iou', 'iou0', 'class', 'mean_x', 'mean_y', 'E', 'E_in', 'E_bd', 
            #     # # # # 'E_rel', 'E_rel_in', 'E_var', 'E_var_in', 'E_var_bd', 'E_var_rel', 'E_var_rel_in', 
            #     # # # # 'D', 'D_in', 'D_bd', 'D_rel', 'D_rel_in', 'D_var', 'D_var_in', 'D_var_bd', 
            #     # # # # 'D_var_rel', 'D_var_rel_in', 'V', 'V_in', 'V_bd', 'V_rel', 'V_rel_in', 'V_var', 
            #     # # # # 'V_var_in', 'V_var_bd', 'V_var_rel', 'V_var_rel_in', 'S', 'S_in', 'S_bd', 'S_rel', 
            #     # # # # 'S_rel_in', 'cprob0', 'cprob1', 'cprob2', 'cprob3', 'cprob4', 'cprob5', 'cprob6', 
            #     # # # # 'cprob7', 'cprob8', 'cprob9', 'cprob10', 'ndist0', 'ndist1', 'ndist2', 'ndist3', 
            #     # # # # 'ndist4', 'ndist5', 'ndist6', 'ndist7', 'ndist8', 'ndist9', 'ndist10']) 



            #     # print(conftouse)          
            #     # print(np.shape(conftouse))

            #     # components2 = 0.*np.ones_like(components)                                            

            #     # for i in range(components.shape[0]):       
            #     #     for j in range(components.shape[1]):
            #     #         #print(components[i, j])            
            #     #         #sadfsadfas
            #     #         if components[i, j] >= 0:       
            #     #             #components[i, j] = conftouse[components[i, j]-1]                                                                                                                               
            #     #             #tempvariable = components[i, j]                     
            #     #             #components[i, j] = conftouse[tempvariable-1]  
            #     #             #components[i, j] = conftouse[0]         
            #     #             #components2[i, j] = conftouse[0]
            #     #             components2[i, j] = conftouse[components[i, j]-1]             
            #     #             #print(conftouse[0])    
            #     #             #print(components[i, j])                                         
            #     #             #print(components2[i, j])  
            #     #             #sadfaszdf
            #     #         else:
            #     #             #components2[i, j] = -conftouse[-components[i, j]-1]
            #     #             components2[i, j] = conftouse[-components[i, j]-1]   
                
            #     # fig = plt.figure()                                     
            #     # plt.imshow(components2)   
            #     # plt.axis('off')        
            #     # plt.colorbar() 
            #     # plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bbbbb.png', bbox_inches='tight')  

            #     # fig = plt.figure()                                               
            #     # plt.imshow(np.where(components2 < 0.40, components2, float("nan")))                                  
            #     # plt.axis('off')                      
            #     # plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bbbbbb.png', bbox_inches='tight')
                
            #     # plt.close('all')                  

            #     # #adfzdf

            #     # #asdfdsa



                
                
            #     # # #conftouse = np.array([4.5625, 9.966666666666667, 8.67379679144385, 25.502202643171806, 71.50544783010157, 54.142857142857146, 59.833333333333336, 76.82, 89.5, 125.1875, 22.92105263157895, 24.0936873747495, 4.0, 69.64589665653496, 99.07979017117614, 19.0, 69.06382978723404, 119.0, 63.213592233009706, 124.57777777777778, 118.22302158273381, 97.875, 4.136363636363637, 16.0, 43.56504854368932, 67.29411764705883, 115.76271186440678, 64.0, 104.875, 7.256038647342995, 24.525252525252526, 42.604166666666664, 126.0])
            #     # # conftouse = np.array(metriccss['mean_y']) 
            #     # # #conftouse = np.max(conftouse) - conftouse      
            #     # # #conftouse /= np.max(conftouse) 
            #     # # conftouse = (conftouse-np.min(conftouse))/(np.max(conftouse)-np.min(conftouse))
            #     # # conftouse = 1. - conftouse

            #     # # #conftouse += np.array([4.4375, 2.466666666666667, 10.358288770053475, 8.101321585903083, 43.61994459833795, 2.642857142857143, 0.6666666666666666, 7.84, 0.0, 1.25, 18.263157894736842, 73.87249498997996, 16.0, 26.445288753799392, 86.37382661512976, 38.5, 41.1063829787234, 42.0, 53.13592233009709, 54.44444444444444, 66.8273381294964, 66.125, 97.86363636363636, 100.33333333333333, 116.05242718446603, 104.70588235294117, 118.09152542372881, 108.0, 117.01785714285714, 121.41545893719807, 124.11111111111111, 124.85416666666667, 122.83333333333333])
            #     # # #conftouse2 = np.array([4.4375, 2.466666666666667, 10.358288770053475, 8.101321585903083, 43.61994459833795, 2.642857142857143, 0.6666666666666666, 7.84, 0.0, 1.25, 18.263157894736842, 73.87249498997996, 16.0, 26.445288753799392, 86.37382661512976, 38.5, 41.1063829787234, 42.0, 53.13592233009709, 54.44444444444444, 66.8273381294964, 66.125, 97.86363636363636, 100.33333333333333, 116.05242718446603, 104.70588235294117, 118.09152542372881, 108.0, 117.01785714285714, 121.41545893719807, 124.11111111111111, 124.85416666666667, 122.83333333333333])
            #     # # conftouse2 = np.array(metriccss['mean_x']) 
            #     # # #conftouse2 /= np.max(conftouse2) 
            #     # # conftouse2 = (conftouse2-np.min(conftouse2))/(np.max(conftouse2)-np.min(conftouse2)) 
            #     # # conftouse += conftouse2
            #     # # conftouse /= 2

            #     # # # conftouse2 = np.array(metriccss['mean_x'])  
            #     # # # #conftouse2 = np.array(metriccss['mean_y'])  
            #     # # # #conftouse2 = (conftouse2-np.min(conftouse2))/(np.max(conftouse2)-np.min(conftouse2))
            #     # # # conftouse = (conftouse2-np.min(conftouse2))/(np.max(conftouse2)-np.min(conftouse2))
            #     # # # #conftouse = 1. - conftouse

            #     # # #conftouse = np.array([4.4375, 2.466666666666667, 10.358288770053475, 8.101321585903083, 43.61994459833795, 2.642857142857143, 0.6666666666666666, 7.84, 0.0, 1.25, 18.263157894736842, 73.87249498997996, 16.0, 26.445288753799392, 86.37382661512976, 38.5, 41.1063829787234, 42.0, 53.13592233009709, 54.44444444444444, 66.8273381294964, 66.125, 97.86363636363636, 100.33333333333333, 116.05242718446603, 104.70588235294117, 118.09152542372881, 108.0, 117.01785714285714, 121.41545893719807, 124.11111111111111, 124.85416666666667, 122.83333333333333])
            #     # # #conftouse += np.array([4.5625, 9.966666666666667, 8.67379679144385, 25.502202643171806, 71.50544783010157, 54.142857142857146, 59.833333333333336, 76.82, 89.5, 125.1875, 22.92105263157895, 24.0936873747495, 4.0, 69.64589665653496, 99.07979017117614, 19.0, 69.06382978723404, 119.0, 63.213592233009706, 124.57777777777778, 118.22302158273381, 97.875, 4.136363636363637, 16.0, 43.56504854368932, 67.29411764705883, 115.76271186440678, 64.0, 104.875, 7.256038647342995, 24.525252525252526, 42.604166666666664, 126.0])
            #     # # #conftouse /= 2

                

            #     # # # conftouse = np.log(conftouse)            

            #     # # # import math   
            #     # # # def sigmoiidd(x):
            #     # # #     return 1 / (1 + math.exp(-x))
                
            #     # # # sigmoidfunction_vv = np.vectorize(sigmoiidd)

            #     # # # #conftouse = sigmoiidd(conftouse)
            #     # # # conftouse = sigmoidfunction_vv(conftouse)

                
                
            #     # # #conftouse = np.array(conftouse) / 100.   
                
            #     # # #conftouse = conftouse / 100.  
            #     # # #conftouse = np.where(conftouse<=1., conftouse, 1.) 



            #     # # # # clip values  
            #     # # conftouse = np.where(conftouse<=1., conftouse, 1.)
            #     # # conftouse = np.where(conftouse>=0., conftouse, 0.)





            #     # # print(conftouse)         
            #     # # print(np.shape(conftouse))

            #     # # components2 = 0.*np.ones_like(components)                                          

            #     # # for i in range(components.shape[0]):       
            #     # #     for j in range(components.shape[1]):
            #     # #         #print(components[i, j])            
            #     # #         #sadfsadfas
            #     # #         if components[i, j] >= 0:       
            #     # #             #components[i, j] = conftouse[components[i, j]-1]                                                                                                                      
            #     # #             #tempvariable = components[i, j]                     
            #     # #             #components[i, j] = conftouse[tempvariable-1]  
            #     # #             #components[i, j] = conftouse[0]         
            #     # #             #components2[i, j] = conftouse[0]
            #     # #             components2[i, j] = conftouse[components[i, j]-1]             
            #     # #             #print(conftouse[0])    
            #     # #             #print(components[i, j])                                         
            #     # #             #print(components2[i, j])  
            #     # #             #sadfaszdf
            #     # #         else:
            #     # #             #components2[i, j] = -conftouse[-components[i, j]-1]
            #     # #             components2[i, j] = conftouse[-components[i, j]-1]   

            #     # # #print(components)                                                                         
            #     # # #print(components.shape)   
                
            #     # # #print(components2)     
            #     # # #print(components2.shape)
                
            #     # # # #adfzs
            #     # # fig = plt.figure()                                     
            #     # # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                                  
            #     # # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())   
            #     # # #plt.imshow(components)   
            #     # # plt.imshow(components2)   
            #     # # plt.axis('off')        
            #     # # plt.colorbar() 
            #     # # #plt.savefig('net_input/img10.png', bbox_inches='tight')    
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10b.png', bbox_inches='tight')                      
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10c.png', bbox_inches='tight')      
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10d.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10e.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10f.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10g.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10h.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10i.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10k.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10l.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10m.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10n.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_1.png', bbox_inches='tight')   
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_3.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_5.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_7.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bb.png', bbox_inches='tight')  
            #     # # plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bbbb.png', bbox_inches='tight')  
            #     # # #plt.savefig('net_input2/img10c.png', bbox_inches='tight')  
            #     # # # #azsdlf

            #     # # fig = plt.figure()                                               
            #     # # #plt.imshow(np.where(components2 < 0.10, components2, float("nan")))                                                                  
            #     # # #plt.imshow(np.where(components2 < 0.20, components2, float("nan")))                                      
            #     # # #plt.imshow(np.where(components2 > 1.10, components2, float("nan")))                                  
            #     # # #plt.imshow(np.where(components2 < 0.40, components2, float("nan")))                                  
            #     # # #plt.imshow(np.where(components2 < 0.41, components2, float("nan")))                                  
            #     # # #plt.imshow(np.where(components2 < 0.30, components2, float("nan")))                                  
            #     # # #plt.imshow(np.where(components2 < 0.31, components2, float("nan")))                                  
            #     # # plt.imshow(np.where(components2 < 0.20, components2, float("nan")))                                  
            #     # # #plt.imshow(np.where(components2 < 0.30, components2, float("nan")))                                  
            #     # # plt.axis('off')                      
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_2.png', bbox_inches='tight')                     
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_4.png', bbox_inches='tight')    
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_6.png', bbox_inches='tight')
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_8.png', bbox_inches='tight')
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9b.png', bbox_inches='tight')
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bbb.png', bbox_inches='tight')
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bbbbb.png', bbox_inches='tight')
            #     # # plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bbbbbb.png', bbox_inches='tight')
                
            #     # # # fig = plt.figure()                                             
            #     # # # plt.imshow(np.where(confmetric > 0.2, confmetric, float("nan")))                              
            #     # # # plt.axis('off')                  
            #     # # # plt.savefig('foldertodownload%s/confmetricgre0.2b.png'%str(vartochange), bbox_inches='tight') 

            #     # # plt.close('all')      

            #     # # asdfsdkfzs

            #     # # #sadfas



            #     # # #components2 = np.where(components > 0, components2, float("nan"))   
            #     # # 0.87025757  
            #     # # 0.86149753

            #     # # #arrayauroc = components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]                  
            #     # # arrayauroc = components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()] 

            #     # # if (kk == 0) and (vartochange == 0):        
            #     # #     #np.save('arrayforauroocc.npy', arrayauroc)
            #     # #     np.save('arrayforauroocc2.npy', arrayauroc)

            #     # #     #np.save('arrayforauroc.npy', arrayauroc)              
            #     # #     #np.save('arrayforauroc2.npy', arrayauroc)
                
            #     # # else:  
            #     # #     #arrayauroc2 = np.load('arrayforauroocc.npy')           
            #     # #     arrayauroc2 = np.load('arrayforauroocc2.npy') 
                    
            #     # #     #arrayauroc2 = np.load('arrayforauroc.npy')             
            #     # #     #arrayauroc2 = np.load('arrayforauroc2.npy') 
                    
            #     # #     arrayauroc = np.concatenate((arrayauroc2, arrayauroc), axis=0)            
                    
            #     # #     #np.save('arrayforauroocc.npy', arrayauroc)        
            #     # #     np.save('arrayforauroocc2.npy', arrayauroc)

            #     # #     #np.save('arrayforauroc.npy', arrayauroc)            
            #     # #     #np.save('arrayforauroc2.npy', arrayauroc) 
            #     # #     # # 0.8749884
                
            #     # #formeanvartochange.append(np.nanmean(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))      
            #     # #formeanvartochange.append(np.nanmean(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanmedian(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()])) 
            #     # #formeanvartochange.append(np.nanmedian(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanstd(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))      
            #     # #formeanvartochange.append(np.nanstd(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.count_nonzero(~np.isnan(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))) 
            #     # #formeanvartochange.append(np.count_nonzero(~np.isnan(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()])))

            #     # #print(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())  
            #     # #print(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())
            #     # #print(np.median(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # #print(np.median(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # #components2 = np.where(components > 0, components2, float("nan")) 

            #     # # print(np.nanmean(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))  
            #     # # print(np.nanmean(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # # print(np.nanmedian(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # # print(np.nanmedian(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # # # 0.9144779904595371
            #     # # # 0.7818230043792824
            #     # # # 0.9086949022718136
            #     # # # 0.8988959333110084                
            #     # # # # 0.6830199735414635   
            #     # # # # 0.12381521563454853 
            #     # # # # 0.9086949022718136
            #     # # # # 0.5877665686876867                

            #     # # arrayauroc = components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()] 
            #     # # #arrayauroc = components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]
                
            #     # # if (kk == 0) and (vartochange == 0): 
            #     # #     np.save('arrayforauroc.npy', arrayauroc) 
            #     # #     #np.save('arrayforauroc2.npy', arrayauroc)
                
            #     # # else: 
            #     # #     arrayauroc2 = np.load('arrayforauroc.npy')
            #     # #     #arrayauroc2 = np.load('arrayforauroc2.npy')
                    
            #     # #     arrayauroc = np.concatenate((arrayauroc2, arrayauroc), axis=0)  
                    
            #     # #     np.save('arrayforauroc.npy', arrayauroc) 
            #     # #     #np.save('arrayforauroc2.npy', arrayauroc) 

            #     # #formeanvartochange.append(np.nanmean(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))      
            #     # #formeanvartochange.append(np.nanmean(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanmedian(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()])) 
            #     # #formeanvartochange.append(np.nanmedian(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanstd(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))      
            #     # #formeanvartochange.append(np.nanstd(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
                
            #     # # 83.149 
            #     # # 81.185

            #     # # 0.926474146782566               
            #     # # 0.8625039948009793  
            #     # # 0.9495011608502848
            #     # # 0.8838974949777696
                
                
                
            #     # #formeanvartochange.append(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())    
            #     # #formeanvartochange.append(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean()) 
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # # # 0.83731733
            #     # # # 0.79144272

            #     # # # 0.96132758  
            #     # # # 0.92786109

            #     # # # 0.95972436         
            #     # # # 0.93700675 

                
                
                
                
            #     # # print(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components90at90[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())   
            #     # # print(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components90at90[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())
            #     # # print(np.median(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components90at90[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # # print(np.median(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components90at90[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

                
                
            #     # #print(outputs3)         
            #     # #print(outputs3.shape) 

            #     # #sadfasdd


                
                
                
            #     # # #import matplotlib.pyplot as plt                             
            #     # # plt.figure()                      
            #     # # #plt.imshow(target_mask)         
            #     # # #plt.imshow(infe)     
            #     # # plt.imshow(components)  
            #     # # #plt.imshow(components2)  
            #     # # #plt.imshow(softmax_vs2)
            #     # # plt.axis('off')     
            #     # # plt.savefig('./Image2_Components.png', bbox_inches='tight')                                 
            #     # # #plt.savefig('./ImageResearch1c.png', bbox_inches='tight')         
            #     # # #plt.savefig('./ImageResearch1d.png', bbox_inches='tight')  
            #     # # #plt.savefig('./ImageResearch1e.png', bbox_inches='tight') 
            #     # # #plt.savefig('./ImageResearch1f.png', bbox_inches='tight')
                
            #     # #asdf                
                
            #     # #sdfsadf
                



                
                
                
                
                
                
                
                
                
            #     # #componenttouse2 = np.ones_like(conftouse, dtype=object)     
            #     # #componenttouse2 = np.ones_like(range(components.max()), dtype=object)  
            #     # componenttouse2 = np.ones_like(range(max(components.max(),-components.min())), dtype=object) 
            #     # #componenttouse2 = np.ones_like(range(components.max()), dtype=object)
            #     # for i in range(len(componenttouse2)):
            #     #     componenttouse2[i] = [0.,]

            #     # for i in range(components.shape[0]):  
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])         
            #     #         if components[i, j] >= 0:  
            #     #             #components[i, j] = conftouse[components[i, j]-1]                                              
            #     #             #tempvariable = components[i, j]              
            #     #             #components[i, j] = conftouse[tempvariable-1] 
            #     #             #components[i, j] = conftouse[0]    
            #     #             #components2[i, j] = conftouse[0]
            #     #             #components2[i, j] = conftouse[components[i, j]-1]
            #     #             if componenttouse2[components[i, j]-1] == [0.,]: 
            #     #                 #print(softmax_vs[i, j]) 
            #     #                 #componenttouse2[components[i, j]-1] = [softmax_vs[i, j],]           
            #     #                 #componenttouse2[components[i, j]-1] = [softmax_vs2[i, j],]   
                                
            #     #                 componenttouse2[components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist()]    
            #     #                 #componenttouse2[components[i, j]-1] = [outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist()]  
                                
            #     #                 #componenttouse2[components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist() * outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist()]
            #     #                 #componenttouse2[components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist() * outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist()]
                            
            #     #             else:
            #     #                 #componenttouse2[components[i, j]-1].append(softmax_vs[i, j])              
            #     #                 #componenttouse2[components[i, j]-1].append(softmax_vs2[i, j])   
                                
            #     #                 componenttouse2[components[i, j]-1].append(outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist()) 
            #     #                 #componenttouse2[components[i, j]-1].append(outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist())
            #     #                 # # -0.09644343 
            #     #                 # # -0.12213121

            #     #                 #componenttouse2[components[i, j]-1].append(outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist() * outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist()) 
            #     #                 # # -0.07944997 
            #     #                 # # -0.09686196 

            #     #                 #componenttouse2[components[i, j]-1].append(outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist() * outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist()) 
                            
            #     #             #componenttouse2[components[i, j]-1].append(softmax_vs[i, j])    
            #     #             #print(conftouse[0])   
            #     #             #print(components[i, j])                                    
            #     #             #print(components2[i, j])      
                        
            #     #         # # (?)
            #     #         # # (?) 
            #     #         # # (?)
            #     #         # else: 
            #     #         #     if componenttouse2[-components[i, j]-1] == [0.,]: 
            #     #         #         #print(softmax_vs[i, j]) 
            #     #         #         #componenttouse2[components[i, j]-1] = [softmax_vs[i, j],]              
            #     #         #         #componenttouse2[components[i, j]-1] = [softmax_vs2[i, j],]   
                                
            #     #         #         componenttouse2[-components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist()]    
            #     #         #         #componenttouse2[components[i, j]-1] = [outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist()]  
                                
            #     #         #         #componenttouse2[components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist() * outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist()]
            #     #         #         #componenttouse2[components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist() * outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist()]
                            
            #     #         #     else:
            #     #         #         #componenttouse2[components[i, j]-1].append(softmax_vs[i, j])                  
            #     #         #         #componenttouse2[components[i, j]-1].append(softmax_vs2[i, j])    
                                
            #     #         #         componenttouse2[-components[i, j]-1].append(outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist())           
            #     #         # # (?)
            #     #         # # (?)
            #     #         # # (?)
            #     #         # # 0.7525941  
            #     #         # # 0.71576604
                
            #     # componenttouse3 = []           
            #     # componenttouse4 = [] 
            #     # componenttouse5 = [] 
            #     # componenttouse6 = [] 
                
            #     # componenttouse5b = []

            #     # componententropyoversegment = []                       

            #     # for componenttouse2a in componenttouse2: 
            #     #     #print(componenttouse2a)  
            #     #     #print(len(componenttouse2a))  
                    
                    
                    
            #     #     # varcounter = 0
            #     #     # varcounter2 = 0
            #     #     # for indexi in range(len(componenttouse2a)):
            #     #     #     if componenttouse2a[indexi] >= 0.9:
            #     #     #         varcounter += 1
            #     #     #     elif componenttouse2a[indexi] <= 0.6:
            #     #     #         varcounter2 += 1
            #     #     # varcounter /= len(componenttouse2a) 
            #     #     # varcounter2 /= len(componenttouse2a)
                    
            #     #     # #print(varcounter)    
            #     #     # #print(varcounter2)  
                    
            #     #     # #print(np.nanmean(componenttouse2a))  
                    
            #     #     # if varcounter >= 0.9:
            #     #     #     componenttouse3.append(1.)
            #     #     # elif varcounter2 >= 0.9:
            #     #     #     componenttouse3.append(0.)
            #     #     # else: 
            #     #     #     componenttouse3.append(np.nanmean(componenttouse2a))     

                    
                    
            #     #     componenttouse3.append(np.nanmean(componenttouse2a)) 
                    
                    
                    
                    
            #     #     #print(componenttouse2a)  
            #     #     #print(len(componenttouse2a))  
            #     #     #print(np.nanmean(componenttouse2a)) 
                    
            #     #     # wsumiinarray = 0.   
            #     #     # for iinarray in range(len(componenttouse2a)):
            #     #     #     rangevalueA = math.floor(componenttouse2a[iinarray]*10.)/10. 
            #     #     #     rangevalueB = math.ceil(componenttouse2a[iinarray]*10.)/10.
            #     #     #     weightwsumiinarray = 0
            #     #     #     for iinarray2 in range(iinarray):
            #     #     #         if (componenttouse2a[iinarray2] > rangevalueA) and (componenttouse2a[iinarray2] < rangevalueB):
            #     #     #             weightwsumiinarray += 1
            #     #     #     for iinarray3 in range(iinarray+1, len(componenttouse2a)):
            #     #     #         if (componenttouse2a[iinarray3] > rangevalueA) and (componenttouse2a[iinarray3] < rangevalueB): 
            #     #     #             weightwsumiinarray += 1
            #     #     #     weightwsumiinarray /= len(componenttouse2a)
            #     #     #     wsumiinarray += componenttouse2a[iinarray] * weightwsumiinarray
                    
            #     #     # print(np.nanmean(componenttouse2a)) 
            #     #     # print(wsumiinarray)



                    
                    
            #     #     #componenttouse3.append(np.mean(componenttouse2a))        
            #     #     #componenttouse3.append(np.nanmean(componenttouse2a))  
            #     #     #componenttouse4.append(np.median(componenttouse2a))
            #     #     componenttouse4.append(np.nanmedian(componenttouse2a))
                    
                    
                    
            #     #     #componenttouse3.append(np.nanmean(componenttouse2a))       
                    
            #     #     # wsumiinarray = 0.   
            #     #     # for iinarray in range(len(componenttouse2a)):
            #     #     #     rangevalueA = math.floor(componenttouse2a[iinarray]*10.)/10.
            #     #     #     rangevalueB = math.ceil(componenttouse2a[iinarray]*10.)/10.
            #     #     #     print(rangevalueA)
            #     #     #     print(rangevalueB)
            #     #     #     asdfzsdf
            #     #     #     weightwsumiinarray = 0
            #     #     #     for iinarray2 in range(iinarray):
            #     #     #         if (componenttouse2a[iinarray2] > rangevalueA) and (componenttouse2a[iinarray2] < rangevalueB):
            #     #     #             weightwsumiinarray += 1
            #     #     #     for iinarray3 in range(iinarray+1, len(componenttouse2a)):
            #     #     #         if (componenttouse2a[iinarray3] > rangevalueA) and (componenttouse2a[iinarray3] < rangevalueB): 
            #     #     #             weightwsumiinarray += 1
            #     #     #     weightwsumiinarray /= len(componenttouse2a)
            #     #     #     wsumiinarray += componenttouse2a[iinarray] * weightwsumiinarray 

            #     #     #componenttouse3.append(np.nanmean(componenttouse2a))  
            #     #     #componenttouse3.append(wsumiinarray)
            #     #     # # 2463.22969496  
            #     #     # # 1996.28738439


                    
            #     #     #componenttouse3.append(np.nanmean(componenttouse2a))      
                    
                    
                    
            #     #     if np.size(componenttouse2a) > 1:   
            #     #         #print(componenttouse2a)    
                        
            #     #         #componententropyoversegment.append(torch.distributions.Categorical(probs = torch.as_tensor(componenttouse2a)).entropy())      
            #     #         #componententropyoversegment.append(torch.distributions.Categorical(probs = torch.Tensor(componenttouse2a)).entropy())  
            #     #         componententropyoversegment.append(np.sum( np.multiply( componenttouse2a, np.log(componenttouse2a+np.finfo(np.float32).eps) ) , axis=-1) / np.log(1.0/np.shape(componenttouse2a)[-1]))    
                        
            #     #         #print(componententropyoversegment)
            #     #         #asdfasdfzs
                    
            #     #     else:   
            #     #         componententropyoversegment.append(float("nan"))   
                    
            #     #     #componententropyoversegment.append(torch.distributions.Categorical(probs = torch.as_tensor(componenttouse2a)).entropy())     
            #     #     #componententropyoversegment.append(torch.distributions.Categorical(probs = torch.Tensor(componenttouse2a)).entropy())  

                    
                    
            #     #     vartoadd = 0       
            #     #     for componenttouse2aa in componenttouse2a:           
            #     #         #if componenttouse2aa >= 0.9:           
            #     #         #if componenttouse2aa >= 0.8: 
            #     #         #if componenttouse2aa >= 0.85:        
            #     #         #if componenttouse2aa >= 0.80:  
            #     #         #if componenttouse2aa >= 0.80:
            #     #         #if componenttouse2aa >= 0.90:
            #     #         #if componenttouse2aa >= 0.70: 
            #     #         #if componenttouse2aa >= 0.90:
            #     #         #if componenttouse2aa >= 0.80:
            #     #         if componenttouse2aa >= 0.90:
            #     #             vartoadd += 1
            #     #             #vartoadd += componenttouse2aa
            #     #             #vartoadd += 1 + componenttouse2aa

            #     #             # # (?) 
            #     #             #vartoadd += componenttouse2aa
            #     #             # # (?)
            #     #     componenttouse5.append(vartoadd)

            #     #     vartoadd2 = 0       
            #     #     for componenttouse2aa in componenttouse2a:         
            #     #         if componenttouse2aa <= 0.60:
            #     #             vartoadd2 += 1
                            
            #     #             # # (?)   
            #     #             #vartoadd2 += componenttouse2aa
            #     #             # # (?)
            #     #     componenttouse5b.append(vartoadd2)

            #     #     componenttouse6.append(len(componenttouse2a))

            #     # for i in range(len(componenttouse5)):  
            #     #     # # (?)
            #     #     componenttouse5[i] /= componenttouse6[i]
            #     #     # # (?)
                    
                    

            #     #     #componenttouse5b[i] /= -componenttouse6[i] 
            #     #     componenttouse5b[i] /= componenttouse6[i]
                   
            #     # del componenttouse6     

            #     # #print(componenttouse5)                          
            #     # #print(np.shape(componenttouse5))      
                
            #     # #print(np.shape(componenttouse5)) 
            #     # #print(components.max())
                
                
                
            #     # # for indexcomponenttouseu55a in range(len(componenttouse5)): 
            #     # #     #if componenttouse5[indexcomponenttouseu55a] > 0.90:
            #     # #     if componenttouse5[indexcomponenttouseu55a] > 0.80:
            #     # #         componenttouse5[indexcomponenttouseu55a] = 1
            #     # #     else:
            #     # #         componenttouse5[indexcomponenttouseu55a] = 0

            #     # # # # 0.11655304   
            #     # # # # 0.04796631
            #     # # # # 0.09375
            #     # # # # 0.03125
            #     # # # # 0.16295985
            #     # # # # 0.10669951
                
                
                
                
                
            #     # #print(componenttouse3)                            
                
            #     # #print(componenttouse4)  

                
                
            #     # #print(np.shape(componententropyoversegment))  
            #     # #print(components.max()) 
                
            #     # #print(componenttouse4)            
            #     # #print(np.shape(componenttouse4))  
                


            #     # componentsmean = 0.*np.ones_like(components)                                                  

            #     # for i in range(components.shape[0]):  
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])               
            #     #         if components[i, j] >= 0:  
            #     #             componentsmean[i, j] = componenttouse3[components[i, j]-1]      
            #     #         else:
            #     #             #print(components[i, j])                        
            #     #             #print(-components[i, j]-1)   
            #     #             #componentsmean[i, j] = -componenttouse3[-components[i, j]-1]

            #     #             # # (?)      
            #     #             # # (?)
            #     #             # # (?)
            #     #             componentsmean[i, j] = -componenttouse3[-components[i, j]-1]  
            #     #             #componentsmean[i, j] = componenttouse3[-components[i, j]-1]
            #     #             # # 0.79042286
            #     #             # # 0.75102869
            #     #             # # (?)
            #     #             # # (?)   
            #     #             # # (?) 

                
                
            #     # # fromlabellss = 0.*np.ones_like(outputss33[vartochange,:,:,:].permute(1,2,0).detach().cpu().numpy())      
            #     # # for iuse in range(fromlabellss.shape[0]): 
            #     # #     for juse in range(fromlabellss.shape[1]):
            #     # #         fromlabellss[iuse, juse, labellss[vartochange, iuse, juse].detach().cpu().numpy()] = 1.
                
            #     # # #print(fromlabellss)     
            #     # # #print(fromlabellss.shape)
            #     # # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # # #_, components = compute_metrics_components(fromlabellss[:,:,:].copy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 

            #     # # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # # #_, components = compute_metrics_components(fromlabellss[:,:,:].copy(), labellss[vartochange,:,:].clone().detach().cpu().numpy())
            #     # # _, componentsb = compute_metrics_components(fromlabellss[:,:,:].copy(), labellss[vartochange,:,:].clone().detach().cpu().numpy())

            #     # # # # outputs3touse = np.where(components > 0, outputs3touse, float("nan"))                                                         
            #     # # #componentsmean = np.where(components > 0, componentsmean, float("nan")) 
            #     # # componentsmean = np.where(componentsb > 0, componentsmean, float("nan"))      
            #     # # # # 0.67781816  
            #     # # # # 0.29641636

                
                
            #     # #componentsmean = np.where(components > 0, componentsmean, float("nan"))       

                
                
            #     # #print(np.nanmean(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))  
            #     # #print(np.nanmean(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #print(np.nanmedian(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #print(np.nanmedian(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # # 0.8790881299787912
            #     # # 0.8132396977263331
            #     # # 0.8613646274651094 
            #     # # 0.8508648423987232                
                
                
                
            #     # # #arrayauroc = outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]             
            #     # # #arrayauroc = outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()] 
                
            #     # # #arrayauroc = componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()] 
            #     # # arrayauroc = componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]
            #     # # # # 0.90231968   
            #     # # # # a 

            #     # # if (kk == 0) and (vartochange == 0):    
            #     # #     #np.save('arrayforauroocc.npy', arrayauroc)       
            #     # #     np.save('arrayforauroocc2.npy', arrayauroc)

            #     # #     #np.save('arrayforauroc.npy', arrayauroc)             
            #     # #     #np.save('arrayforauroc2.npy', arrayauroc)
                
            #     # # else:  
            #     # #     #arrayauroc2 = np.load('arrayforauroocc.npy')   
            #     # #     arrayauroc2 = np.load('arrayforauroocc2.npy')
                    
            #     # #     #arrayauroc2 = np.load('arrayforauroc.npy')           
            #     # #     #arrayauroc2 = np.load('arrayforauroc2.npy') 
                    
            #     # #     arrayauroc = np.concatenate((arrayauroc2, arrayauroc), axis=0)     
                    
            #     # #     #np.save('arrayforauroocc.npy', arrayauroc)     
            #     # #     np.save('arrayforauroocc2.npy', arrayauroc)

            #     # #     #np.save('arrayforauroc.npy', arrayauroc)                                              
            #     # #     #np.save('arrayforauroc2.npy', arrayauroc)      

            #     # #formeanvartochange.append(np.nanmean(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))      
            #     # #formeanvartochange.append(np.nanmean(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanmedian(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()])) 
            #     # #formeanvartochange.append(np.nanmedian(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanstd(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))      
            #     # #formeanvartochange.append(np.nanstd(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.count_nonzero(~np.isnan(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))) 
            #     # #formeanvartochange.append(np.count_nonzero(~np.isnan(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()])))

                
                
            #     # #formeanvartochange.append(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0].mean())   
            #     # #formeanvartochange.append(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0].mean())
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0])) 
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0]))

            #     # # # 0.92817041    
            #     # # # 0.89029398

            #     # # # Mean:           
            #     # # # 0.90298937
            #     # # # 0.90049725 

                
                
                
                
                
                
            #     # componentsmedian = 0.*np.ones_like(components)                                        

            #     # for i in range(components.shape[0]):       
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])               
            #     #         if components[i, j] >= 0:  
            #     #             componentsmedian[i, j] = componenttouse4[components[i, j]-1]       
            #     #         else:
            #     #             #componentsmedian[i, j] = -componenttouse4[-components[i, j]-1] 

            #     #             # # (?)
            #     #             componentsmedian[i, j] = -componenttouse4[-components[i, j]-1]    
            #     #             #componentsmedian[i, j] = componenttouse4[-components[i, j]-1]
            #     #             # # 0.81813379 
            #     #             # # 0.77486875
            #     #             # # (?) 

            #     # # print(componentsmedian[infe==gt_masks][components[infe==gt_masks]>=0.].mean())                           
            #     # # print(componentsmedian[infe!=gt_masks][components[infe!=gt_masks]>=0.].mean())    
            #     # # print(np.median(componentsmedian[infe==gt_masks][components[infe==gt_masks]>=0.]))             
            #     # # print(np.median(componentsmedian[infe!=gt_masks][components[infe!=gt_masks]>=0.])) 

            #     # #print(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0].mean()) 
            #     # #print(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0].mean())
            #     # #print(np.median(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0]))
            #     # #print(np.median(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0]))

            #     # # # outputs3touse = np.where(components > 0, outputs3touse, float("nan"))                                                                    
            #     # #componentsmedian = np.where(components > 0, componentsmedian, float("nan"))     

            #     # # #arrayauroc = componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]
            #     # # arrayauroc = componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]
                
            #     # # if (kk == 0) and (vartochange == 0): 
            #     # #     #np.save('arrayforauroc.npy', arrayauroc) 
            #     # #     np.save('arrayforauroc2.npy', arrayauroc)
                
            #     # # else: 
            #     # #     #arrayauroc2 = np.load('arrayforauroc.npy')
            #     # #     arrayauroc2 = np.load('arrayforauroc2.npy')
                    
            #     # #     arrayauroc = np.concatenate((arrayauroc2, arrayauroc), axis=0)  
                    
            #     # #     #np.save('arrayforauroc.npy', arrayauroc)  
            #     # #     np.save('arrayforauroc2.npy', arrayauroc) 

            #     # #formeanvartochange.append(np.nanmean(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))          
            #     # #formeanvartochange.append(np.nanmean(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()])) 
            #     # #formeanvartochange.append(np.nanmedian(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()])) 
            #     # #formeanvartochange.append(np.nanmedian(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanstd(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))         
            #     # #formeanvartochange.append(np.nanstd(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
                
            #     # #formeanvartochange.append(len(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))           
            #     # #formeanvartochange.append(np.count_nonzero(~np.isnan(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))) 
            #     # #formeanvartochange.append(np.count_nonzero(~np.isnan(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))) 
                
            #     # #formeanvartochange.append(np.nanmean(outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))        
            #     # #formeanvartochange.append(np.nanmean(outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()])) 
            #     # #formeanvartochange.append(np.nanmedian(outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanmedian(outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanstd(outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))      
            #     # #formeanvartochange.append(np.nanstd(outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()])) 
            #     # #formeanvartochange.append(np.count_nonzero(~np.isnan(outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))) 
            #     # #formeanvartochange.append(np.count_nonzero(~np.isnan(outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()])))

            #     # #formeanvartochange.append(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0].mean())    
            #     # #formeanvartochange.append(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0].mean()) 
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0])) 
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0]))

            #     # # # Median:                       
            #     # # # 0.95357752  
            #     # # # 0.94796602

                
                
                
                
                
                
                
                
                
                
            #     # #sdz

            #     # #asd

            #     # components90at90 = 0.*np.ones_like(components)                                             

            #     # for i in range(components.shape[0]):          
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])                              
            #     #         if components[i, j] >= 0:   
            #     #             #if componenttouse5[components[i, j]-1] >= 0.9:                
            #     #             #if componenttouse5[components[i, j]-1] >= 0.8:  
            #     #             #if componenttouse5[components[i, j]-1] >= 0.85: 
            #     #             # if componenttouse5[components[i, j]-1] >= 0.80:
            #     #             #     components90at90[i, j] = componenttouse5[components[i, j]-1]                      
            #     #             # else:  
            #     #             #     components90at90[i, j] = -componenttouse5[components[i, j]-1] 
            #     #             components90at90[i, j] = componenttouse5[components[i, j]-1]   
                        
            #     #         else:
            #     #             # # (?) 
            #     #             # # (?) 
            #     #             #if componenttouse5[-components[i, j]-1] != 0:
            #     #             #    components90at90[i, j] = -componenttouse5[-components[i, j]-1]
            #     #             #else:
            #     #             #    components90at90[i, j] = -1
            #     #             # # (?)   
            #     #             # # (?) 
                            
            #     #             components90at90[i, j] = -componenttouse5[-components[i, j]-1]   
            #     #             #components90at90[i, j] = componenttouse5[-components[i, j]-1] 

                
                
            #     # #print(components90at90)                                  
            #     # #print(components90at90.shape)    

            #     # # # 8         
            #     # outputs3g = components90at90   
            #     # # # this is per segment (and not per pixel), i.e. 1 value per segment                                 
                
            #     # #adfzs
            #     # fig = plt.figure()                                   
            #     # plt.imshow(render_s2_as_rgb( images[vartochange,:3,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1] ), interpolation="nearest")     
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/input.png'%str(vartochange), bbox_inches='tight')   

            #     # fig = plt.figure()                                     
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                                        
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())     
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())                      
            #     # #plt.imshow(outputs3b[0, :, :].detach().cpu().numpy())      
            #     # #plt.imshow(outputs3c[0, :, :].detach().cpu().numpy())   
            #     # plt.imshow(1.-np.where(outputs3g > 0, outputs3g, 0), cmap='Reds')  
            #     # #plt.imshow(1.-outputs3g, cmap='Reds') 
            #     # plt.colorbar() 
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/greaterthan90.png'%str(vartochange), bbox_inches='tight')    
            #     # #plt.savefig('foldertodownload/greaterthan60.png', bbox_inches='tight')    
            #     # #azsdlf
                
                
                
                
                
            #     # fig = plt.figure()                                      
            #     # #plt.imshow(1.-np.where(componentsmean > 0, componentsmean, 0), cmap='Reds')      
            #     # plt.imshow(1.-componentsmean, cmap='Reds')    
            #     # plt.colorbar() 
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/meanoversegment.png'%str(vartochange), bbox_inches='tight')    
                
            #     # fig = plt.figure()                                       
            #     # #plt.imshow(1.-np.where(componentsmedian > 0, componentsmean, 0), cmap='Reds')           
            #     # plt.imshow(1.-componentsmedian, cmap='Reds')      
            #     # plt.colorbar() 
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/medianoversegment.png'%str(vartochange), bbox_inches='tight')     
                
            #     # #asfasdfas

                

                
                
            #     # # # componententropyoversegment        
            #     # componentsentropy = 0.*np.ones_like(components)                                      
            #     # for i in range(components.shape[0]):          
            #     #     for j in range(components.shape[1]): 
            #     #         if components[i, j] >= 0:   
            #     #             componentsentropy[i, j] = componententropyoversegment[components[i, j]-1]   
            #     #         else:
            #     #             componentsentropy[i, j] = -componententropyoversegment[-components[i, j]-1]  
            #     #             #componentsentropy[i, j] = componententropyoversegment[-components[i, j]-1]

            #     # #print(componentsentropy)         
            #     # #print(componentsentropy.shape)
                
            #     # # # 9 
            #     # outputs3h = componentsentropy
            #     # # # this is per segment (and not per pixel), i.e. 1 value per segment        
                
            #     # #adfzs
            #     # fig = plt.figure()                                     
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                         
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())   
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())          
            #     # #plt.imshow(outputs3b[0, :, :].detach().cpu().numpy())   
            #     # #plt.imshow(outputs3c[0, :, :].detach().cpu().numpy())   
            #     # plt.imshow((100.-np.where(outputs3h > 0, outputs3h, 0))/100., cmap='Reds')  
            #     # #plt.imshow((100.-outputs3h)/100., cmap='Reds') 
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/entroversegm.png'%str(vartochange), bbox_inches='tight')   
            #     # #azsdlf
                
                
                
                
                
            #     # components90at90b = 0.*np.ones_like(components)                                             

            #     # for i in range(components.shape[0]):          
            #     #     for j in range(components.shape[1]): 
            #     #         if components[i, j] >= 0:   
            #     #             components90at90b[i, j] = componenttouse5b[components[i, j]-1]     
                        
            #     #         else:
            #     #             #print(componenttouse5b[-components[i, j]-1])       
            #     #             #if componenttouse5b[-components[i, j]-1] != 0:
            #     #             #    components90at90b[i, j] = -componenttouse5b[-components[i, j]-1] 
            #     #             #else:
            #     #             #    components90at90b[i, j] = -1
                            
            #     #             components90at90b[i, j] = -componenttouse5b[-components[i, j]-1]    
            #     #             #components90at90b[i, j] = componenttouse5b[-components[i, j]-1]

                
                
            #     # # # 10         
            #     # outputs3i = components90at90b
            #     # # # this is per segment (and not per pixel), i.e. 1 value per segment            

            #     # #adfzs
            #     # fig = plt.figure()                                     
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                                 
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())    
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())             
            #     # #plt.imshow(outputs3b[0, :, :].detach().cpu().numpy())   
            #     # #plt.imshow(outputs3c[0, :, :].detach().cpu().numpy())   
            #     # plt.imshow(np.where(outputs3i > 0, outputs3i, 0), cmap='Reds')
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/lessthan60.png'%str(vartochange), bbox_inches='tight')   
            #     # #azsdlf
                
            #     # # #adfzs
            #     # # fig = plt.figure()                                      
            #     # # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                          
            #     # # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())    
            #     # # #plt.imshow(components)   
            #     # # #plt.imshow(components2)   
            #     # # #plt.imshow(outputs3touse)   
            #     # # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())          
            #     # # #plt.imshow(outputs3b[0, :, :].detach().cpu().numpy())   
            #     # # #plt.imshow(outputs3c[0, :, :].detach().cpu().numpy())   
            #     # # #plt.imshow(1.-np.where(outputs3g > 0, outputs3g, 0) + (100.-np.where(outputs3h > 0, outputs3h, 0))/100. + np.where(outputs3i > 0, outputs3i, 0), cmap='Reds')
                
            #     # # def sigmoidfunction(x):
            #     # #   return 1 / (1 + math.exp(-x))
            #     # # sigmoidfunction_v = np.vectorize(sigmoidfunction)
            #     # # #plt.imshow(sigmoidfunction_v(1.-np.where(outputs3g > 0, outputs3g, 0) + (100.-np.where(outputs3h > 0, outputs3h, 0))/100. + np.where(outputs3i > 0, outputs3i, 0)), cmap='Reds') 
            #     # # plt.imshow(1.-sigmoidfunction_v(1.-np.where(outputs3g > 0, outputs3g, 0) + (100.-np.where(outputs3h > 0, outputs3h, 0))/100. + np.where(outputs3i > 0, outputs3i, 0))) 

            #     # # plt.colorbar()
            #     # # plt.axis('off')         
            #     # # plt.savefig('net_input/img18.png', bbox_inches='tight') 
            #     # # #azsdlf

                
                
                
                
            #     # #asdfsadf
            #     # fig = plt.figure()                                      
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                     
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())  
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(1.-outputs3[0, :, :].detach().cpu().numpy(), cmap='Reds')      
            #     # plt.imshow(1.-np.where(components > 0, outputs3[vartochange, :, :].detach().cpu().numpy(), 0), cmap='Reds')        
            #     # #plt.imshow(1.-outputs3[vartochange, :, :].detach().cpu().numpy(), cmap='Reds')      
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/softmaxprob.png'%str(vartochange), bbox_inches='tight')    
            #     # #asdfsadf

            #     # fig = plt.figure()                                      
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                     
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())    
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(1.-outputs3a[0, :, :].detach().cpu().numpy(), cmap='Reds')     
            #     # plt.imshow(1.-np.where(components > 0, outputs3a[vartochange, :, :].detach().cpu().numpy(), 0), cmap='Reds')      
            #     # #plt.imshow(1.-outputs3a[vartochange, :, :].detach().cpu().numpy(), cmap='Reds')      
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/diffDz.png'%str(vartochange), bbox_inches='tight')   
            #     # #asdfasdf

            #     # fig = plt.figure()                                     
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                     
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())    
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())     
            #     # #plt.imshow(-outputs3b[0, :, :].detach().cpu().numpy(), cmap='Reds')    
            #     # #plt.imshow(-outputs3b[vartochange, :, :].detach().cpu().numpy(), cmap='Reds')    
            #     # plt.imshow(-np.where(components > 0, outputs3b[vartochange, :, :].detach().cpu().numpy(), -1), cmap='Reds')     
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/entrperpixel.png'%str(vartochange), bbox_inches='tight')  
            #     # #asdfasdf

            #     # fig = plt.figure()                                     
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                   
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())   
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())        
            #     # #plt.imshow(outputs3b[0, :, :].detach().cpu().numpy())    
            #     # #plt.imshow(1.-outputs3c[0, :, :].detach().cpu().numpy(), cmap='Reds')    
            #     # plt.imshow(1.-np.where(components > 0, outputs3c[vartochange, :, :].detach().cpu().numpy(), 0), cmap='Reds')      
            #     # #plt.imshow(1.-outputs3c[vartochange, :, :].detach().cpu().numpy(), cmap='Reds')      
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/diff1and3.png'%str(vartochange), bbox_inches='tight')   
            #     # #sadfszdf
                
                

            #     # fig = plt.figure()                                     
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                   
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())    
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())        
            #     # #plt.imshow(outputs3b[0, :, :].detach().cpu().numpy())   
            #     # #plt.imshow(outputs3c[0, :, :].detach().cpu().numpy())   
            #     # plt.imshow(-outputs3d[vartochange, :, :].detach().cpu().numpy(), cmap='Reds')   
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/logits1st.png'%str(vartochange), bbox_inches='tight')  
                
            #     # fig = plt.figure()                                     
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                         
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())    
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())          
            #     # #plt.imshow(outputs3b[0, :, :].detach().cpu().numpy())   
            #     # #plt.imshow(outputs3c[0, :, :].detach().cpu().numpy())   
            #     # plt.imshow(-outputs3e[vartochange, :, :].detach().cpu().numpy(), cmap='Reds')   
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/logitsdiffDz.png'%str(vartochange), bbox_inches='tight')   
                
            #     # fig = plt.figure()                                      
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                           
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())    
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())              
            #     # #plt.imshow(outputs3b[0, :, :].detach().cpu().numpy())   
            #     # #plt.imshow(outputs3c[0, :, :].detach().cpu().numpy())   
            #     # plt.imshow(-outputs3f[vartochange, :, :].detach().cpu().numpy(), cmap='Reds')     
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/logitsdiff1and3.png'%str(vartochange), bbox_inches='tight')    
            #     # #sadfasdf

                
                
            #     # #fig = plt.figure()                                          
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())        
            #     # #plt.axis('off')          
            #     # #plt.savefig('foldertodownload%s/groundtruth.png'%str(vartochange), bbox_inches='tight')      
                
            #     # #fig = plt.figure()                                       
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())         
            #     # #plt.axis('off')           
            #     # #plt.savefig('foldertodownload%s/prediction.png'%str(vartochange), bbox_inches='tight')      
                
            #     # #print(labellss[vartochange, :, :].detach().cpu().numpy().shape) 
                
                
                
            #     #print(images.shape)       
            #     #print(labels.shape) 
            #     #print(outputs.shape)
                
            #     #sadfasdf

            #     # torch.Size([42, 10, 128, 128])
            #     # torch.Size([42, 1, 128, 128])
            #     # torch.Size([42, 1, 128, 128])

                
                
                
                
                
            #     # def ade_palette(): 
            #     #     """ADE20K palette that maps each class to RGB values. """
            #     #     return [[0, 100, 0], [255, 187, 34], [255, 255, 76], [240, 150, 255],
            #     #             [250, 0, 0], [180, 180, 180], [240, 240, 240], [0, 100, 200],
            #     #             [0, 150, 160], [0, 207, 117], [250, 230, 160]]

            #     # color_seg = np.zeros((labellss[vartochange, :, :].detach().cpu().numpy().shape[0],
            #     #                     labellss[vartochange, :, :].detach().cpu().numpy().shape[1], 3), dtype=np.uint8) # height, width, 3 

            #     # palette = np.array(ade_palette())
            #     # for label, color in enumerate(palette):
            #     #     color_seg[labellss[vartochange, :, :].detach().cpu().numpy() == label, :] = color

            #     #print(color_seg.shape)     

            #     fig = plt.figure()                                            
            #     #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())             
            #     #plt.imshow(color_seg)   
            #     #plt.imshow(color_seg) 
            #     plt.imshow(labels[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())   
            #     plt.axis('off')          
            #     plt.colorbar()
            #     plt.savefig('foldertodownload%s/groundtruth.png'%str(vartochange), bbox_inches='tight')      
                
                

            #     # color_seg = np.zeros((theoutput[vartochange, :, :].detach().cpu().numpy().shape[0],
            #     #                     theoutput[vartochange, :, :].detach().cpu().numpy().shape[1], 3), dtype=np.uint8) # # height, width, 3     

            #     # palette = np.array(ade_palette())
            #     # for label, color in enumerate(palette):
            #     #     color_seg[theoutput[vartochange, :, :].detach().cpu().numpy() == label, :] = color
                
            #     fig = plt.figure()                                              
            #     #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())                      
            #     #plt.imshow(color_seg)         
            #     #plt.imshow(color_seg)  
            #     plt.imshow(outputs[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())  
            #     plt.axis('off')           
            #     plt.colorbar()
            #     plt.savefig('foldertodownload%s/prediction.png'%str(vartochange), bbox_inches='tight')

            #     fig = plt.figure()                                                 
            #     #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())                                         
            #     #plt.imshow(color_seg)             
            #     #plt.imshow(color_seg)  
            #     #plt.imshow(outputs[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())    
            #     plt.imshow(outputs2[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())  
            #     plt.axis('off')           
            #     plt.colorbar()
            #     plt.savefig('foldertodownload%s/predictioonn.png'%str(vartochange), bbox_inches='tight')



            #     fig = plt.figure()                                               
            #     #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                                                                         
            #     #plt.imshow(color_seg)      
            #     #plt.imshow(color_seg) 
            #     #plt.imshow(labels[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())    
            #     #plt.axis('off')           
            #     #plt.colorbar()
            #     #plt.savefig('foldertodownload%s/groundtruth.png'%str(vartochange), bbox_inches='tight')      
            #     #plt.imshow(outputs[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())  
            #     #plt.imshow(((labels[vartochange, :, :, :]-outputs[vartochange, :, :, :])**2).permute(1, 2, 0).detach().cpu().numpy())  
            #     # # torch.abs(   
            #     #plt.imshow((torch.abs(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :])).permute(1, 2, 0).detach().cpu().numpy())  
            #     #plt.imshow(torch.abs(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :]).permute(1, 2, 0).detach().cpu().numpy())  
            #     # # torch.pow    
            #     #plt.imshow(torch.pow(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :], 2).permute(1, 2, 0).detach().cpu().numpy())  
            #     #plt.imshow(torch.pow(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :], 2).permute(1, 2, 0).detach().cpu().numpy())  
            #     #plt.imshow(torch.pow(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :], 2).permute(1, 2, 0).detach().cpu().numpy())  
            #     #plt.imshow(torch.pow(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :], 2).permute(1, 2, 0).detach().cpu().numpy())  
            #     # # torch.sqrt     
            #     #plt.imshow(torch.sqrt(torch.pow(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :], 2)).permute(1, 2, 0).detach().cpu().numpy())  
            #     #plt.imshow(torch.sqrt(torch.pow(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :], 2)).permute(1, 2, 0).detach().cpu().numpy())  
            #     #plt.imshow(np.where(torch.sqrt(torch.pow(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :], 2)).permute(1, 2, 0).detach().cpu().numpy()>0.2, torch.sqrt(torch.pow(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :], 2)).permute(1, 2, 0).detach().cpu().numpy(), float("nan")))   
            #     plt.imshow(np.where(torch.sqrt(torch.pow(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :], 2)).permute(1, 2, 0).detach().cpu().numpy()>0.1, torch.sqrt(torch.pow(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :], 2)).permute(1, 2, 0).detach().cpu().numpy(), float("nan")))   
                
            #     #plt.imshow(np.where(torch.pow(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :], 2).permute(1, 2, 0).detach().cpu().numpy()>0.1, torch.pow(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :], 2).permute(1, 2, 0).detach().cpu().numpy(), float("nan")))  
            #     #plt.imshow(np.where(torch.pow(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :], 2).permute(1, 2, 0).detach().cpu().numpy()>0.2, torch.pow(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :], 2).permute(1, 2, 0).detach().cpu().numpy(), float("nan")))  
            #     #plt.imshow(np.where(torch.pow(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :], 2).permute(1, 2, 0).detach().cpu().numpy()>0.64, torch.pow(labels[vartochange, :, :, :]-outputs[vartochange, :, :, :], 2).permute(1, 2, 0).detach().cpu().numpy(), float("nan")))  
                
            #     plt.axis('off')            
            #     plt.colorbar()
            #     plt.savefig('foldertodownload%s/predictionn.png'%str(vartochange), bbox_inches='tight')

            #     fig = plt.figure()                                                     
            #     #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())                                                                                                                                                                                                              
            #     #plt.imshow(color_seg)                   
            #     #plt.imshow(color_seg)   
            #     #plt.imshow(outputs[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())      
            #     #plt.imshow(outputs2[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())   
            #     #plt.imshow(outputs2[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())  
            #     # # plt.imshow(np.where(confmetric > 0.2, confmetric, float("nan")))  
            #     plt.imshow(np.where(outputs2[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy()<0.2, outputs2[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy(), float("nan")))   
            #     #plt.imshow(outputs2[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())    
            #     plt.axis('off')            
            #     plt.colorbar()
            #     plt.savefig('foldertodownload%s/predictionnn.png'%str(vartochange), bbox_inches='tight') 



            #     fig = plt.figure()                                                 
            #     #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())                                                                          
            #     #plt.imshow(color_seg)               
            #     #plt.imshow(color_seg)  
            #     #plt.imshow(outputs[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())     
            #     #plt.imshow(outputs2[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())  
            #     #plt.imshow(outputs2[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())  
            #     plt.imshow(1.-outputs2[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())  
            #     plt.axis('off')           
            #     plt.colorbar()
            #     plt.savefig('foldertodownload%s/predictioonn2.png'%str(vartochange), bbox_inches='tight')

            #     fig = plt.figure()                                                 
            #     #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())                                                                                  
            #     #plt.imshow(color_seg)               
            #     #plt.imshow(color_seg)  
            #     #plt.imshow(outputs[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())     
            #     #plt.imshow(outputs2[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())  
            #     #plt.imshow(outputs2[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())  
            #     plt.imshow(torch.abs(outputs[vartochange, :, :, :]-labels[vartochange, :, :, :]).permute(1, 2, 0).detach().cpu().numpy())  
            #     #plt.imshow(((outputs[vartochange, :, :, :]-labels[vartochange, :, :, :])**2).permute(1, 2, 0).detach().cpu().numpy())  
            #     plt.axis('off')           
            #     plt.colorbar()
            #     plt.savefig('foldertodownload%s/predictioonn3.png'%str(vartochange), bbox_inches='tight')

            #     fig = plt.figure()                                                   
            #     #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())                                                                                                                                                                       
            #     #plt.imshow(color_seg)                  
            #     #plt.imshow(color_seg)  
            #     #plt.imshow(outputs[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())       
            #     #plt.imshow(outputs2[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())  
            #     #plt.imshow(outputs2[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())  
            #     #plt.imshow(1.-outputs2[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())   
            #     #plt.imshow(1.-outputs2[vartochange, :, :, :].permute(1, 2, 0).detach().cpu().numpy())  
            #     plt.imshow(torch.abs((1.-outputs2[vartochange, :, :, :])-torch.abs(outputs[vartochange, :, :, :]-labels[vartochange, :, :, :])).permute(1, 2, 0).detach().cpu().numpy())  
            #     #plt.imshow(torch.abs((1.-outputs2[vartochange, :, :, :])-((outputs[vartochange, :, :, :]-labels[vartochange, :, :, :])**2)).permute(1, 2, 0).detach().cpu().numpy())  
            #     plt.axis('off')            
            #     plt.colorbar()
            #     plt.savefig('foldertodownload%s/predictioonn4.png'%str(vartochange), bbox_inches='tight')

            #     print('')   
            #     print(torch.abs((1.-outputs2[vartochange, :, :, :])-torch.abs(outputs[vartochange, :, :, :]-labels[vartochange, :, :, :])).mean())
            #     print(torch.abs((1.-outputs2[vartochange, :, :, :])-torch.abs(outputs[vartochange, :, :, :]-labels[vartochange, :, :, :])).median())
            #     # # IoU  
            #     # # minimum

            
            
            #     #asdfsadf  

            #     #ads

            #     #adfsdzf




                
                
                
            #     # # def rgba(r, g, b):            
            #     # #     a = 1 - np.min(r, np.min(g, b)) / 255  
            #     # #     return [255 + (r - 255) / a, 255 + (g - 255) / a, 255 + (b - 255) / a, a]      

            #     # fig = plt.figure()                                                         
            #     # plt.imshow(color_seg)  
            #     # plt.axis('off')           
            #     # #colors = [im.cmap(im.norm(value)) for value in values] 
            #     # #labels7 = ['Tree cover', 'Shrubland', 'Grassland', 'Cropland', 'Built-up', 'Bare/ sparse vegetation', 'Snow and ice', 'Permanent water bodies', 'Herbaceous wetland', 'Mangroves', 'Moss and lichen'] 
            #     # #import matplotlib.patches as mpatches  
            #     # #patches = [mpatches.Patch(color=rgba(palette[i][0],palette[i][1],palette[i][2]), label7=labels7[i]) for i in range(11)]          
            #     # #plt.legend(handles=patches, loc=4, borderaxespad=0.)
            #     # plt.hist([], color='g', label="Tree cover") 
            #     # plt.hist([], color='gold', label="Shrubland")
            #     # plt.hist([], color='yellow', label="Grassland")
            #     # plt.hist([], color='plum', label="Cropland")
            #     # plt.hist([], color='red', label="Built-up")
            #     # plt.hist([], color='silver', label="Bare/ sparse vegetation")
            #     # plt.hist([], color='white', label="Snow and ice")
            #     # plt.hist([], color='blue', label="Permanent water bodies")
            #     # plt.hist([], color='teal', label="Herbaceous wetland")
            #     # plt.hist([], color='lightgreen', label="Mangroves") 
            #     # plt.hist([], color='khaki', label="Moss and lichen") 
            #     # plt.legend(loc='lower left')  
            #     # plt.savefig('foldertodownload%s/prediction2.png'%str(vartochange), bbox_inches='tight')   

            #     # asdfas



                
                
                
            #     # from metrics import compute_metrics_components    
            #     # #print(outputss33.shape)
            #     # #print(labellss.shape)
                
            #     # # fromlabellss = 0.*np.ones_like(outputss33[vartochange,:,:,:].permute(1,2,0).detach().cpu().numpy())        
            #     # # for iuse in range(fromlabellss.shape[0]): 
            #     # #     for juse in range(fromlabellss.shape[1]):
            #     # #         fromlabellss[iuse, juse, labellss[vartochange, iuse, juse].detach().cpu().numpy()] = 1.
                
            #     # #print(fromlabellss)    
            #     # #print(fromlabellss.shape)
            #     # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # #_, components = compute_metrics_components(fromlabellss[:,:,:].copy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 

            #     # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # metriccss, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # #_, components = compute_metrics_components(fromlabellss[:,:,:].copy(), labellss[vartochange,:,:].clone().detach().cpu().numpy())  

            #     # #print(metriccss)
            #     # #print(metriccss.keys())  
            #     # #asdkfzsdf

            #     # #print(metriccss.keys())  
            #     # #asdfzsdf

            #     # # # D_rel 
            #     # # # D_var
            #     # # # D_var_in
            #     # # # (?)
                
                
                
                
            #     # #metriccss['E_in']                        
            #     # #metriccss['D_in']    
            #     # #metriccss['V_in']

            #     # # # dict_keys(['iou', 'iou0', 'class', 'mean_x', 'mean_y', 'E', 'E_in', 'E_bd', 
            #     # # # 'E_rel', 'E_rel_in', 'E_var', 'E_var_in', 'E_var_bd', 'E_var_rel', 'E_var_rel_in', 
            #     # # # 'D', 'D_in', 'D_bd', 'D_rel', 'D_rel_in', 'D_var', 'D_var_in', 'D_var_bd', 
            #     # # # 'D_var_rel', 'D_var_rel_in', 'V', 'V_in', 'V_bd', 'V_rel', 'V_rel_in', 'V_var', 
            #     # # # 'V_var_in', 'V_var_bd', 'V_var_rel', 'V_var_rel_in', 'S', 'S_in', 'S_bd', 'S_rel', 
            #     # # # 'S_rel_in', 'cprob0', 'cprob1', 'cprob2', 'cprob3', 'cprob4', 'cprob5', 'cprob6', 
            #     # # # 'cprob7', 'cprob8', 'cprob9', 'cprob10', 'ndist0', 'ndist1', 'ndist2', 'ndist3', 
            #     # # # 'ndist4', 'ndist5', 'ndist6', 'ndist7', 'ndist8', 'ndist9', 'ndist10'])  

            #     # #sdfsa



            #     # # print('')    
            #     # # print(components)    
            #     # # print(components.shape) 
                
            #     # # print(components.max())                  
            #     # # print(-components.min())  
                
            #     # #asdfas


                
            #     # # from metrics import compute_IoU_component 
            #     # # theIoUs = compute_IoU_component(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy(), True)
            #     # # #theIoUs = compute_IoU_component(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), labellss[vartochange,:,:].clone().detach().cpu().numpy(), True)

            #     # # print(theIoUs)
            #     # # print(theIoUs.shape)

            #     # #asdfzsf

                
                
            #     # # #import matplotlib.pyplot as plt                                     
            #     # # plt.figure()                      
            #     # # #plt.imshow(target_mask)             
            #     # # #plt.imshow(infe)      
            #     # # plt.imshow(components)  
            #     # # #plt.imshow(components2) 
            #     # # #plt.imshow(softmax_vs2)
            #     # # plt.axis('off')     
            #     # # plt.colorbar()
            #     # # plt.savefig('/Data/temp/phileo17022024/phileo-bench/net_input2/img11.png', bbox_inches='tight')

                
                
            #     # # # (?)                                          
            #     # # if vartochange == 0:       
            #     # #     continue
            #     # # elif vartochange == 1:     
            #     # #    continue 
            #     # # # (?)
                
            #     # #conftouse = np.array(metriccss['mean_y'])       
            #     # #conftouse = np.array(metriccss['D'])   
                
                
                
            #     # conftouse = np.array(metriccss['D_in']) 
            #     # conftouse = (conftouse-np.min(conftouse))/(np.max(conftouse)-np.min(conftouse))
            #     # conftouse = 1. - conftouse

            #     # conftoussee = np.array(metriccss['E_in']) 
            #     # conftoussee = (conftoussee-np.min(conftoussee))/(np.max(conftoussee)-np.min(conftoussee))
            #     # conftoussee = 1. - conftoussee
            #     # conftouse += conftoussee 
                
            #     # conftoussee = np.array(metriccss['V_in'])  
            #     # conftoussee = (conftoussee-np.min(conftoussee))/(np.max(conftoussee)-np.min(conftoussee))
            #     # conftoussee = 1. - conftoussee
            #     # conftouse += conftoussee
                
            #     # conftouse /= 3     

                
                
            #     # # conftouse = conftouse = np.array(metriccss['D_var']) 
            #     # # conftouse = (conftouse-np.min(conftouse))/(np.max(conftouse)-np.min(conftouse))

            #     # # #metriccss['E_in']                                                            
            #     # # #metriccss['D_in']      
            #     # # #metriccss['V_in'] 

            #     # # # # dict_keys(['iou', 'iou0', 'class', 'mean_x', 'mean_y', 'E', 'E_in', 'E_bd', 
            #     # # # # 'E_rel', 'E_rel_in', 'E_var', 'E_var_in', 'E_var_bd', 'E_var_rel', 'E_var_rel_in', 
            #     # # # # 'D', 'D_in', 'D_bd', 'D_rel', 'D_rel_in', 'D_var', 'D_var_in', 'D_var_bd', 
            #     # # # # 'D_var_rel', 'D_var_rel_in', 'V', 'V_in', 'V_bd', 'V_rel', 'V_rel_in', 'V_var', 
            #     # # # # 'V_var_in', 'V_var_bd', 'V_var_rel', 'V_var_rel_in', 'S', 'S_in', 'S_bd', 'S_rel', 
            #     # # # # 'S_rel_in', 'cprob0', 'cprob1', 'cprob2', 'cprob3', 'cprob4', 'cprob5', 'cprob6', 
            #     # # # # 'cprob7', 'cprob8', 'cprob9', 'cprob10', 'ndist0', 'ndist1', 'ndist2', 'ndist3', 
            #     # # # # 'ndist4', 'ndist5', 'ndist6', 'ndist7', 'ndist8', 'ndist9', 'ndist10']) 



            #     # print(conftouse)          
            #     # print(np.shape(conftouse))

            #     # components2 = 0.*np.ones_like(components)                                            

            #     # for i in range(components.shape[0]):       
            #     #     for j in range(components.shape[1]):
            #     #         #print(components[i, j])            
            #     #         #sadfsadfas
            #     #         if components[i, j] >= 0:       
            #     #             #components[i, j] = conftouse[components[i, j]-1]                                                                                                                               
            #     #             #tempvariable = components[i, j]                     
            #     #             #components[i, j] = conftouse[tempvariable-1]  
            #     #             #components[i, j] = conftouse[0]         
            #     #             #components2[i, j] = conftouse[0]
            #     #             components2[i, j] = conftouse[components[i, j]-1]             
            #     #             #print(conftouse[0])    
            #     #             #print(components[i, j])                                         
            #     #             #print(components2[i, j])  
            #     #             #sadfaszdf
            #     #         else:
            #     #             #components2[i, j] = -conftouse[-components[i, j]-1]
            #     #             components2[i, j] = conftouse[-components[i, j]-1]   
                
            #     # fig = plt.figure()                                     
            #     # plt.imshow(components2)   
            #     # plt.axis('off')        
            #     # plt.colorbar() 
            #     # plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bbbbb.png', bbox_inches='tight')  

            #     # fig = plt.figure()                                               
            #     # plt.imshow(np.where(components2 < 0.40, components2, float("nan")))                                  
            #     # plt.axis('off')                      
            #     # plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bbbbbb.png', bbox_inches='tight')
                
            #     # plt.close('all')                  

            #     # #adfzdf

            #     # #asdfdsa



                
                
            #     # # #conftouse = np.array([4.5625, 9.966666666666667, 8.67379679144385, 25.502202643171806, 71.50544783010157, 54.142857142857146, 59.833333333333336, 76.82, 89.5, 125.1875, 22.92105263157895, 24.0936873747495, 4.0, 69.64589665653496, 99.07979017117614, 19.0, 69.06382978723404, 119.0, 63.213592233009706, 124.57777777777778, 118.22302158273381, 97.875, 4.136363636363637, 16.0, 43.56504854368932, 67.29411764705883, 115.76271186440678, 64.0, 104.875, 7.256038647342995, 24.525252525252526, 42.604166666666664, 126.0])
            #     # # conftouse = np.array(metriccss['mean_y']) 
            #     # # #conftouse = np.max(conftouse) - conftouse      
            #     # # #conftouse /= np.max(conftouse) 
            #     # # conftouse = (conftouse-np.min(conftouse))/(np.max(conftouse)-np.min(conftouse))
            #     # # conftouse = 1. - conftouse

            #     # # #conftouse += np.array([4.4375, 2.466666666666667, 10.358288770053475, 8.101321585903083, 43.61994459833795, 2.642857142857143, 0.6666666666666666, 7.84, 0.0, 1.25, 18.263157894736842, 73.87249498997996, 16.0, 26.445288753799392, 86.37382661512976, 38.5, 41.1063829787234, 42.0, 53.13592233009709, 54.44444444444444, 66.8273381294964, 66.125, 97.86363636363636, 100.33333333333333, 116.05242718446603, 104.70588235294117, 118.09152542372881, 108.0, 117.01785714285714, 121.41545893719807, 124.11111111111111, 124.85416666666667, 122.83333333333333])
            #     # # #conftouse2 = np.array([4.4375, 2.466666666666667, 10.358288770053475, 8.101321585903083, 43.61994459833795, 2.642857142857143, 0.6666666666666666, 7.84, 0.0, 1.25, 18.263157894736842, 73.87249498997996, 16.0, 26.445288753799392, 86.37382661512976, 38.5, 41.1063829787234, 42.0, 53.13592233009709, 54.44444444444444, 66.8273381294964, 66.125, 97.86363636363636, 100.33333333333333, 116.05242718446603, 104.70588235294117, 118.09152542372881, 108.0, 117.01785714285714, 121.41545893719807, 124.11111111111111, 124.85416666666667, 122.83333333333333])
            #     # # conftouse2 = np.array(metriccss['mean_x']) 
            #     # # #conftouse2 /= np.max(conftouse2) 
            #     # # conftouse2 = (conftouse2-np.min(conftouse2))/(np.max(conftouse2)-np.min(conftouse2)) 
            #     # # conftouse += conftouse2
            #     # # conftouse /= 2

            #     # # # conftouse2 = np.array(metriccss['mean_x'])  
            #     # # # #conftouse2 = np.array(metriccss['mean_y'])  
            #     # # # #conftouse2 = (conftouse2-np.min(conftouse2))/(np.max(conftouse2)-np.min(conftouse2))
            #     # # # conftouse = (conftouse2-np.min(conftouse2))/(np.max(conftouse2)-np.min(conftouse2))
            #     # # # #conftouse = 1. - conftouse

            #     # # #conftouse = np.array([4.4375, 2.466666666666667, 10.358288770053475, 8.101321585903083, 43.61994459833795, 2.642857142857143, 0.6666666666666666, 7.84, 0.0, 1.25, 18.263157894736842, 73.87249498997996, 16.0, 26.445288753799392, 86.37382661512976, 38.5, 41.1063829787234, 42.0, 53.13592233009709, 54.44444444444444, 66.8273381294964, 66.125, 97.86363636363636, 100.33333333333333, 116.05242718446603, 104.70588235294117, 118.09152542372881, 108.0, 117.01785714285714, 121.41545893719807, 124.11111111111111, 124.85416666666667, 122.83333333333333])
            #     # # #conftouse += np.array([4.5625, 9.966666666666667, 8.67379679144385, 25.502202643171806, 71.50544783010157, 54.142857142857146, 59.833333333333336, 76.82, 89.5, 125.1875, 22.92105263157895, 24.0936873747495, 4.0, 69.64589665653496, 99.07979017117614, 19.0, 69.06382978723404, 119.0, 63.213592233009706, 124.57777777777778, 118.22302158273381, 97.875, 4.136363636363637, 16.0, 43.56504854368932, 67.29411764705883, 115.76271186440678, 64.0, 104.875, 7.256038647342995, 24.525252525252526, 42.604166666666664, 126.0])
            #     # # #conftouse /= 2

                

            #     # # # conftouse = np.log(conftouse)            

            #     # # # import math   
            #     # # # def sigmoiidd(x):
            #     # # #     return 1 / (1 + math.exp(-x))
                
            #     # # # sigmoidfunction_vv = np.vectorize(sigmoiidd)

            #     # # # #conftouse = sigmoiidd(conftouse)
            #     # # # conftouse = sigmoidfunction_vv(conftouse)

                
                
            #     # # #conftouse = np.array(conftouse) / 100.   
                
            #     # # #conftouse = conftouse / 100.  
            #     # # #conftouse = np.where(conftouse<=1., conftouse, 1.) 



            #     # # # # clip values  
            #     # # conftouse = np.where(conftouse<=1., conftouse, 1.)
            #     # # conftouse = np.where(conftouse>=0., conftouse, 0.)





            #     # # print(conftouse)         
            #     # # print(np.shape(conftouse))

            #     # # components2 = 0.*np.ones_like(components)                                          

            #     # # for i in range(components.shape[0]):       
            #     # #     for j in range(components.shape[1]):
            #     # #         #print(components[i, j])            
            #     # #         #sadfsadfas
            #     # #         if components[i, j] >= 0:       
            #     # #             #components[i, j] = conftouse[components[i, j]-1]                                                                                                                      
            #     # #             #tempvariable = components[i, j]                     
            #     # #             #components[i, j] = conftouse[tempvariable-1]  
            #     # #             #components[i, j] = conftouse[0]         
            #     # #             #components2[i, j] = conftouse[0]
            #     # #             components2[i, j] = conftouse[components[i, j]-1]             
            #     # #             #print(conftouse[0])    
            #     # #             #print(components[i, j])                                         
            #     # #             #print(components2[i, j])  
            #     # #             #sadfaszdf
            #     # #         else:
            #     # #             #components2[i, j] = -conftouse[-components[i, j]-1]
            #     # #             components2[i, j] = conftouse[-components[i, j]-1]   

            #     # # #print(components)                                                                         
            #     # # #print(components.shape)   
                
            #     # # #print(components2)     
            #     # # #print(components2.shape)
                
            #     # # # #adfzs
            #     # # fig = plt.figure()                                     
            #     # # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                                  
            #     # # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())   
            #     # # #plt.imshow(components)   
            #     # # plt.imshow(components2)   
            #     # # plt.axis('off')        
            #     # # plt.colorbar() 
            #     # # #plt.savefig('net_input/img10.png', bbox_inches='tight')    
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10b.png', bbox_inches='tight')                      
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10c.png', bbox_inches='tight')      
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10d.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10e.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10f.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10g.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10h.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10i.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10k.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10l.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10m.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10n.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_1.png', bbox_inches='tight')   
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_3.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_5.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_7.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9.png', bbox_inches='tight')  
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bb.png', bbox_inches='tight')  
            #     # # plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bbbb.png', bbox_inches='tight')  
            #     # # #plt.savefig('net_input2/img10c.png', bbox_inches='tight')  
            #     # # # #azsdlf

            #     # # fig = plt.figure()                                               
            #     # # #plt.imshow(np.where(components2 < 0.10, components2, float("nan")))                                                                  
            #     # # #plt.imshow(np.where(components2 < 0.20, components2, float("nan")))                                      
            #     # # #plt.imshow(np.where(components2 > 1.10, components2, float("nan")))                                  
            #     # # #plt.imshow(np.where(components2 < 0.40, components2, float("nan")))                                  
            #     # # #plt.imshow(np.where(components2 < 0.41, components2, float("nan")))                                  
            #     # # #plt.imshow(np.where(components2 < 0.30, components2, float("nan")))                                  
            #     # # #plt.imshow(np.where(components2 < 0.31, components2, float("nan")))                                  
            #     # # plt.imshow(np.where(components2 < 0.20, components2, float("nan")))                                  
            #     # # #plt.imshow(np.where(components2 < 0.30, components2, float("nan")))                                  
            #     # # plt.axis('off')                      
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_2.png', bbox_inches='tight')                     
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_4.png', bbox_inches='tight')    
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_6.png', bbox_inches='tight')
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_8.png', bbox_inches='tight')
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9b.png', bbox_inches='tight')
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bbb.png', bbox_inches='tight')
            #     # # #plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bbbbb.png', bbox_inches='tight')
            #     # # plt.savefig('/home/ndionelis/Datatemp/homendionelis/Datatemp/phileo17022024/phileo-bench/net_input2/img10_9bbbbbb.png', bbox_inches='tight')
                
            #     # # # fig = plt.figure()                                             
            #     # # # plt.imshow(np.where(confmetric > 0.2, confmetric, float("nan")))                              
            #     # # # plt.axis('off')                  
            #     # # # plt.savefig('foldertodownload%s/confmetricgre0.2b.png'%str(vartochange), bbox_inches='tight') 

            #     # # plt.close('all')      

            #     # # asdfsdkfzs

            #     # # #sadfas



            #     # # #components2 = np.where(components > 0, components2, float("nan"))   
            #     # # 0.87025757  
            #     # # 0.86149753

            #     # # #arrayauroc = components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]                  
            #     # # arrayauroc = components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()] 

            #     # # if (kk == 0) and (vartochange == 0):        
            #     # #     #np.save('arrayforauroocc.npy', arrayauroc)
            #     # #     np.save('arrayforauroocc2.npy', arrayauroc)

            #     # #     #np.save('arrayforauroc.npy', arrayauroc)              
            #     # #     #np.save('arrayforauroc2.npy', arrayauroc)
                
            #     # # else:  
            #     # #     #arrayauroc2 = np.load('arrayforauroocc.npy')           
            #     # #     arrayauroc2 = np.load('arrayforauroocc2.npy') 
                    
            #     # #     #arrayauroc2 = np.load('arrayforauroc.npy')             
            #     # #     #arrayauroc2 = np.load('arrayforauroc2.npy') 
                    
            #     # #     arrayauroc = np.concatenate((arrayauroc2, arrayauroc), axis=0)            
                    
            #     # #     #np.save('arrayforauroocc.npy', arrayauroc)        
            #     # #     np.save('arrayforauroocc2.npy', arrayauroc)

            #     # #     #np.save('arrayforauroc.npy', arrayauroc)            
            #     # #     #np.save('arrayforauroc2.npy', arrayauroc) 
            #     # #     # # 0.8749884
                
            #     # #formeanvartochange.append(np.nanmean(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))      
            #     # #formeanvartochange.append(np.nanmean(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanmedian(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()])) 
            #     # #formeanvartochange.append(np.nanmedian(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanstd(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))      
            #     # #formeanvartochange.append(np.nanstd(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.count_nonzero(~np.isnan(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))) 
            #     # #formeanvartochange.append(np.count_nonzero(~np.isnan(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()])))

            #     # #print(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())  
            #     # #print(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())
            #     # #print(np.median(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # #print(np.median(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # #components2 = np.where(components > 0, components2, float("nan")) 

            #     # # print(np.nanmean(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))  
            #     # # print(np.nanmean(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # # print(np.nanmedian(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # # print(np.nanmedian(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # # # 0.9144779904595371
            #     # # # 0.7818230043792824
            #     # # # 0.9086949022718136
            #     # # # 0.8988959333110084                
            #     # # # # 0.6830199735414635   
            #     # # # # 0.12381521563454853 
            #     # # # # 0.9086949022718136
            #     # # # # 0.5877665686876867                

            #     # # arrayauroc = components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()] 
            #     # # #arrayauroc = components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]
                
            #     # # if (kk == 0) and (vartochange == 0): 
            #     # #     np.save('arrayforauroc.npy', arrayauroc) 
            #     # #     #np.save('arrayforauroc2.npy', arrayauroc)
                
            #     # # else: 
            #     # #     arrayauroc2 = np.load('arrayforauroc.npy')
            #     # #     #arrayauroc2 = np.load('arrayforauroc2.npy')
                    
            #     # #     arrayauroc = np.concatenate((arrayauroc2, arrayauroc), axis=0)  
                    
            #     # #     np.save('arrayforauroc.npy', arrayauroc) 
            #     # #     #np.save('arrayforauroc2.npy', arrayauroc) 

            #     # #formeanvartochange.append(np.nanmean(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))      
            #     # #formeanvartochange.append(np.nanmean(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanmedian(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()])) 
            #     # #formeanvartochange.append(np.nanmedian(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanstd(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))      
            #     # #formeanvartochange.append(np.nanstd(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
                
            #     # # 83.149 
            #     # # 81.185

            #     # # 0.926474146782566               
            #     # # 0.8625039948009793  
            #     # # 0.9495011608502848
            #     # # 0.8838974949777696
                
                
                
            #     # #formeanvartochange.append(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())    
            #     # #formeanvartochange.append(components2[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean()) 
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # # # 0.83731733
            #     # # # 0.79144272

            #     # # # 0.96132758  
            #     # # # 0.92786109

            #     # # # 0.95972436         
            #     # # # 0.93700675 

                
                
                
                
            #     # # print(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components90at90[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())   
            #     # # print(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components90at90[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())
            #     # # print(np.median(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components90at90[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # # print(np.median(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components90at90[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

                
                
            #     # #print(outputs3)         
            #     # #print(outputs3.shape) 

            #     # #sadfasdd


                
                
                
            #     # # #import matplotlib.pyplot as plt                             
            #     # # plt.figure()                      
            #     # # #plt.imshow(target_mask)         
            #     # # #plt.imshow(infe)     
            #     # # plt.imshow(components)  
            #     # # #plt.imshow(components2)  
            #     # # #plt.imshow(softmax_vs2)
            #     # # plt.axis('off')     
            #     # # plt.savefig('./Image2_Components.png', bbox_inches='tight')                                 
            #     # # #plt.savefig('./ImageResearch1c.png', bbox_inches='tight')         
            #     # # #plt.savefig('./ImageResearch1d.png', bbox_inches='tight')  
            #     # # #plt.savefig('./ImageResearch1e.png', bbox_inches='tight') 
            #     # # #plt.savefig('./ImageResearch1f.png', bbox_inches='tight')
                
            #     # #asdf                
                
            #     # #sdfsadf
                



                
                
                
                
                
                
                
                
                
            #     # #componenttouse2 = np.ones_like(conftouse, dtype=object)     
            #     # #componenttouse2 = np.ones_like(range(components.max()), dtype=object)  
            #     # componenttouse2 = np.ones_like(range(max(components.max(),-components.min())), dtype=object) 
            #     # #componenttouse2 = np.ones_like(range(components.max()), dtype=object)
            #     # for i in range(len(componenttouse2)):
            #     #     componenttouse2[i] = [0.,]

            #     # for i in range(components.shape[0]):  
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])         
            #     #         if components[i, j] >= 0:  
            #     #             #components[i, j] = conftouse[components[i, j]-1]                                              
            #     #             #tempvariable = components[i, j]              
            #     #             #components[i, j] = conftouse[tempvariable-1] 
            #     #             #components[i, j] = conftouse[0]    
            #     #             #components2[i, j] = conftouse[0]
            #     #             #components2[i, j] = conftouse[components[i, j]-1]
            #     #             if componenttouse2[components[i, j]-1] == [0.,]: 
            #     #                 #print(softmax_vs[i, j]) 
            #     #                 #componenttouse2[components[i, j]-1] = [softmax_vs[i, j],]           
            #     #                 #componenttouse2[components[i, j]-1] = [softmax_vs2[i, j],]   
                                
            #     #                 componenttouse2[components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist()]    
            #     #                 #componenttouse2[components[i, j]-1] = [outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist()]  
                                
            #     #                 #componenttouse2[components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist() * outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist()]
            #     #                 #componenttouse2[components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist() * outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist()]
                            
            #     #             else:
            #     #                 #componenttouse2[components[i, j]-1].append(softmax_vs[i, j])              
            #     #                 #componenttouse2[components[i, j]-1].append(softmax_vs2[i, j])   
                                
            #     #                 componenttouse2[components[i, j]-1].append(outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist()) 
            #     #                 #componenttouse2[components[i, j]-1].append(outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist())
            #     #                 # # -0.09644343 
            #     #                 # # -0.12213121

            #     #                 #componenttouse2[components[i, j]-1].append(outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist() * outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist()) 
            #     #                 # # -0.07944997 
            #     #                 # # -0.09686196 

            #     #                 #componenttouse2[components[i, j]-1].append(outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist() * outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist()) 
                            
            #     #             #componenttouse2[components[i, j]-1].append(softmax_vs[i, j])    
            #     #             #print(conftouse[0])   
            #     #             #print(components[i, j])                                    
            #     #             #print(components2[i, j])      
                        
            #     #         # # (?)
            #     #         # # (?) 
            #     #         # # (?)
            #     #         # else: 
            #     #         #     if componenttouse2[-components[i, j]-1] == [0.,]: 
            #     #         #         #print(softmax_vs[i, j]) 
            #     #         #         #componenttouse2[components[i, j]-1] = [softmax_vs[i, j],]              
            #     #         #         #componenttouse2[components[i, j]-1] = [softmax_vs2[i, j],]   
                                
            #     #         #         componenttouse2[-components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist()]    
            #     #         #         #componenttouse2[components[i, j]-1] = [outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist()]  
                                
            #     #         #         #componenttouse2[components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist() * outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist()]
            #     #         #         #componenttouse2[components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist() * outputs3b[vartochange, i, j].clone().detach().cpu().numpy().tolist()]
                            
            #     #         #     else:
            #     #         #         #componenttouse2[components[i, j]-1].append(softmax_vs[i, j])                  
            #     #         #         #componenttouse2[components[i, j]-1].append(softmax_vs2[i, j])    
                                
            #     #         #         componenttouse2[-components[i, j]-1].append(outputs3[vartochange, i, j].clone().detach().cpu().numpy().tolist())           
            #     #         # # (?)
            #     #         # # (?)
            #     #         # # (?)
            #     #         # # 0.7525941  
            #     #         # # 0.71576604
                
            #     # componenttouse3 = []           
            #     # componenttouse4 = [] 
            #     # componenttouse5 = [] 
            #     # componenttouse6 = [] 
                
            #     # componenttouse5b = []

            #     # componententropyoversegment = []                       

            #     # for componenttouse2a in componenttouse2: 
            #     #     #print(componenttouse2a)  
            #     #     #print(len(componenttouse2a))  
                    
                    
                    
            #     #     # varcounter = 0
            #     #     # varcounter2 = 0
            #     #     # for indexi in range(len(componenttouse2a)):
            #     #     #     if componenttouse2a[indexi] >= 0.9:
            #     #     #         varcounter += 1
            #     #     #     elif componenttouse2a[indexi] <= 0.6:
            #     #     #         varcounter2 += 1
            #     #     # varcounter /= len(componenttouse2a) 
            #     #     # varcounter2 /= len(componenttouse2a)
                    
            #     #     # #print(varcounter)    
            #     #     # #print(varcounter2)  
                    
            #     #     # #print(np.nanmean(componenttouse2a))  
                    
            #     #     # if varcounter >= 0.9:
            #     #     #     componenttouse3.append(1.)
            #     #     # elif varcounter2 >= 0.9:
            #     #     #     componenttouse3.append(0.)
            #     #     # else: 
            #     #     #     componenttouse3.append(np.nanmean(componenttouse2a))     

                    
                    
            #     #     componenttouse3.append(np.nanmean(componenttouse2a)) 
                    
                    
                    
                    
            #     #     #print(componenttouse2a)  
            #     #     #print(len(componenttouse2a))  
            #     #     #print(np.nanmean(componenttouse2a)) 
                    
            #     #     # wsumiinarray = 0.   
            #     #     # for iinarray in range(len(componenttouse2a)):
            #     #     #     rangevalueA = math.floor(componenttouse2a[iinarray]*10.)/10. 
            #     #     #     rangevalueB = math.ceil(componenttouse2a[iinarray]*10.)/10.
            #     #     #     weightwsumiinarray = 0
            #     #     #     for iinarray2 in range(iinarray):
            #     #     #         if (componenttouse2a[iinarray2] > rangevalueA) and (componenttouse2a[iinarray2] < rangevalueB):
            #     #     #             weightwsumiinarray += 1
            #     #     #     for iinarray3 in range(iinarray+1, len(componenttouse2a)):
            #     #     #         if (componenttouse2a[iinarray3] > rangevalueA) and (componenttouse2a[iinarray3] < rangevalueB): 
            #     #     #             weightwsumiinarray += 1
            #     #     #     weightwsumiinarray /= len(componenttouse2a)
            #     #     #     wsumiinarray += componenttouse2a[iinarray] * weightwsumiinarray
                    
            #     #     # print(np.nanmean(componenttouse2a)) 
            #     #     # print(wsumiinarray)



                    
                    
            #     #     #componenttouse3.append(np.mean(componenttouse2a))        
            #     #     #componenttouse3.append(np.nanmean(componenttouse2a))  
            #     #     #componenttouse4.append(np.median(componenttouse2a))
            #     #     componenttouse4.append(np.nanmedian(componenttouse2a))
                    
                    
                    
            #     #     #componenttouse3.append(np.nanmean(componenttouse2a))       
                    
            #     #     # wsumiinarray = 0.   
            #     #     # for iinarray in range(len(componenttouse2a)):
            #     #     #     rangevalueA = math.floor(componenttouse2a[iinarray]*10.)/10.
            #     #     #     rangevalueB = math.ceil(componenttouse2a[iinarray]*10.)/10.
            #     #     #     print(rangevalueA)
            #     #     #     print(rangevalueB)
            #     #     #     asdfzsdf
            #     #     #     weightwsumiinarray = 0
            #     #     #     for iinarray2 in range(iinarray):
            #     #     #         if (componenttouse2a[iinarray2] > rangevalueA) and (componenttouse2a[iinarray2] < rangevalueB):
            #     #     #             weightwsumiinarray += 1
            #     #     #     for iinarray3 in range(iinarray+1, len(componenttouse2a)):
            #     #     #         if (componenttouse2a[iinarray3] > rangevalueA) and (componenttouse2a[iinarray3] < rangevalueB): 
            #     #     #             weightwsumiinarray += 1
            #     #     #     weightwsumiinarray /= len(componenttouse2a)
            #     #     #     wsumiinarray += componenttouse2a[iinarray] * weightwsumiinarray 

            #     #     #componenttouse3.append(np.nanmean(componenttouse2a))  
            #     #     #componenttouse3.append(wsumiinarray)
            #     #     # # 2463.22969496  
            #     #     # # 1996.28738439


                    
            #     #     #componenttouse3.append(np.nanmean(componenttouse2a))      
                    
                    
                    
            #     #     if np.size(componenttouse2a) > 1:   
            #     #         #print(componenttouse2a)    
                        
            #     #         #componententropyoversegment.append(torch.distributions.Categorical(probs = torch.as_tensor(componenttouse2a)).entropy())      
            #     #         #componententropyoversegment.append(torch.distributions.Categorical(probs = torch.Tensor(componenttouse2a)).entropy())  
            #     #         componententropyoversegment.append(np.sum( np.multiply( componenttouse2a, np.log(componenttouse2a+np.finfo(np.float32).eps) ) , axis=-1) / np.log(1.0/np.shape(componenttouse2a)[-1]))    
                        
            #     #         #print(componententropyoversegment)
            #     #         #asdfasdfzs
                    
            #     #     else:   
            #     #         componententropyoversegment.append(float("nan"))   
                    
            #     #     #componententropyoversegment.append(torch.distributions.Categorical(probs = torch.as_tensor(componenttouse2a)).entropy())     
            #     #     #componententropyoversegment.append(torch.distributions.Categorical(probs = torch.Tensor(componenttouse2a)).entropy())  

                    
                    
            #     #     vartoadd = 0       
            #     #     for componenttouse2aa in componenttouse2a:           
            #     #         #if componenttouse2aa >= 0.9:           
            #     #         #if componenttouse2aa >= 0.8: 
            #     #         #if componenttouse2aa >= 0.85:        
            #     #         #if componenttouse2aa >= 0.80:  
            #     #         #if componenttouse2aa >= 0.80:
            #     #         #if componenttouse2aa >= 0.90:
            #     #         #if componenttouse2aa >= 0.70: 
            #     #         #if componenttouse2aa >= 0.90:
            #     #         #if componenttouse2aa >= 0.80:
            #     #         if componenttouse2aa >= 0.90:
            #     #             vartoadd += 1
            #     #             #vartoadd += componenttouse2aa
            #     #             #vartoadd += 1 + componenttouse2aa

            #     #             # # (?) 
            #     #             #vartoadd += componenttouse2aa
            #     #             # # (?)
            #     #     componenttouse5.append(vartoadd)

            #     #     vartoadd2 = 0       
            #     #     for componenttouse2aa in componenttouse2a:         
            #     #         if componenttouse2aa <= 0.60:
            #     #             vartoadd2 += 1
                            
            #     #             # # (?)   
            #     #             #vartoadd2 += componenttouse2aa
            #     #             # # (?)
            #     #     componenttouse5b.append(vartoadd2)

            #     #     componenttouse6.append(len(componenttouse2a))

            #     # for i in range(len(componenttouse5)):  
            #     #     # # (?)
            #     #     componenttouse5[i] /= componenttouse6[i]
            #     #     # # (?)
                    
                    

            #     #     #componenttouse5b[i] /= -componenttouse6[i] 
            #     #     componenttouse5b[i] /= componenttouse6[i]
                   
            #     # del componenttouse6     

            #     # #print(componenttouse5)                          
            #     # #print(np.shape(componenttouse5))      
                
            #     # #print(np.shape(componenttouse5)) 
            #     # #print(components.max())
                
                
                
            #     # # for indexcomponenttouseu55a in range(len(componenttouse5)): 
            #     # #     #if componenttouse5[indexcomponenttouseu55a] > 0.90:
            #     # #     if componenttouse5[indexcomponenttouseu55a] > 0.80:
            #     # #         componenttouse5[indexcomponenttouseu55a] = 1
            #     # #     else:
            #     # #         componenttouse5[indexcomponenttouseu55a] = 0

            #     # # # # 0.11655304   
            #     # # # # 0.04796631
            #     # # # # 0.09375
            #     # # # # 0.03125
            #     # # # # 0.16295985
            #     # # # # 0.10669951
                
                
                
                
                
            #     # #print(componenttouse3)                            
                
            #     # #print(componenttouse4)  

                
                
            #     # #print(np.shape(componententropyoversegment))  
            #     # #print(components.max()) 
                
            #     # #print(componenttouse4)            
            #     # #print(np.shape(componenttouse4))  
                


            #     # componentsmean = 0.*np.ones_like(components)                                                  

            #     # for i in range(components.shape[0]):  
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])               
            #     #         if components[i, j] >= 0:  
            #     #             componentsmean[i, j] = componenttouse3[components[i, j]-1]      
            #     #         else:
            #     #             #print(components[i, j])                        
            #     #             #print(-components[i, j]-1)   
            #     #             #componentsmean[i, j] = -componenttouse3[-components[i, j]-1]

            #     #             # # (?)      
            #     #             # # (?)
            #     #             # # (?)
            #     #             componentsmean[i, j] = -componenttouse3[-components[i, j]-1]  
            #     #             #componentsmean[i, j] = componenttouse3[-components[i, j]-1]
            #     #             # # 0.79042286
            #     #             # # 0.75102869
            #     #             # # (?)
            #     #             # # (?)   
            #     #             # # (?) 

                
                
            #     # # fromlabellss = 0.*np.ones_like(outputss33[vartochange,:,:,:].permute(1,2,0).detach().cpu().numpy())      
            #     # # for iuse in range(fromlabellss.shape[0]): 
            #     # #     for juse in range(fromlabellss.shape[1]):
            #     # #         fromlabellss[iuse, juse, labellss[vartochange, iuse, juse].detach().cpu().numpy()] = 1.
                
            #     # # #print(fromlabellss)     
            #     # # #print(fromlabellss.shape)
            #     # # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # # #_, components = compute_metrics_components(fromlabellss[:,:,:].copy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 

            #     # # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # # #_, components = compute_metrics_components(fromlabellss[:,:,:].copy(), labellss[vartochange,:,:].clone().detach().cpu().numpy())
            #     # # _, componentsb = compute_metrics_components(fromlabellss[:,:,:].copy(), labellss[vartochange,:,:].clone().detach().cpu().numpy())

            #     # # # # outputs3touse = np.where(components > 0, outputs3touse, float("nan"))                                                         
            #     # # #componentsmean = np.where(components > 0, componentsmean, float("nan")) 
            #     # # componentsmean = np.where(componentsb > 0, componentsmean, float("nan"))      
            #     # # # # 0.67781816  
            #     # # # # 0.29641636

                
                
            #     # #componentsmean = np.where(components > 0, componentsmean, float("nan"))       

                
                
            #     # #print(np.nanmean(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))  
            #     # #print(np.nanmean(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #print(np.nanmedian(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #print(np.nanmedian(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # # 0.8790881299787912
            #     # # 0.8132396977263331
            #     # # 0.8613646274651094 
            #     # # 0.8508648423987232                
                
                
                
            #     # # #arrayauroc = outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]             
            #     # # #arrayauroc = outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()] 
                
            #     # # #arrayauroc = componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()] 
            #     # # arrayauroc = componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]
            #     # # # # 0.90231968   
            #     # # # # a 

            #     # # if (kk == 0) and (vartochange == 0):    
            #     # #     #np.save('arrayforauroocc.npy', arrayauroc)       
            #     # #     np.save('arrayforauroocc2.npy', arrayauroc)

            #     # #     #np.save('arrayforauroc.npy', arrayauroc)             
            #     # #     #np.save('arrayforauroc2.npy', arrayauroc)
                
            #     # # else:  
            #     # #     #arrayauroc2 = np.load('arrayforauroocc.npy')   
            #     # #     arrayauroc2 = np.load('arrayforauroocc2.npy')
                    
            #     # #     #arrayauroc2 = np.load('arrayforauroc.npy')           
            #     # #     #arrayauroc2 = np.load('arrayforauroc2.npy') 
                    
            #     # #     arrayauroc = np.concatenate((arrayauroc2, arrayauroc), axis=0)     
                    
            #     # #     #np.save('arrayforauroocc.npy', arrayauroc)     
            #     # #     np.save('arrayforauroocc2.npy', arrayauroc)

            #     # #     #np.save('arrayforauroc.npy', arrayauroc)                                              
            #     # #     #np.save('arrayforauroc2.npy', arrayauroc)      

            #     # #formeanvartochange.append(np.nanmean(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))      
            #     # #formeanvartochange.append(np.nanmean(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanmedian(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()])) 
            #     # #formeanvartochange.append(np.nanmedian(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanstd(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))      
            #     # #formeanvartochange.append(np.nanstd(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.count_nonzero(~np.isnan(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))) 
            #     # #formeanvartochange.append(np.count_nonzero(~np.isnan(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()])))

                
                
            #     # #formeanvartochange.append(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0].mean())   
            #     # #formeanvartochange.append(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0].mean())
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0])) 
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0]))

            #     # # # 0.92817041    
            #     # # # 0.89029398

            #     # # # Mean:           
            #     # # # 0.90298937
            #     # # # 0.90049725 

                
                
                
                
                
                
            #     # componentsmedian = 0.*np.ones_like(components)                                        

            #     # for i in range(components.shape[0]):       
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])               
            #     #         if components[i, j] >= 0:  
            #     #             componentsmedian[i, j] = componenttouse4[components[i, j]-1]       
            #     #         else:
            #     #             #componentsmedian[i, j] = -componenttouse4[-components[i, j]-1] 

            #     #             # # (?)
            #     #             componentsmedian[i, j] = -componenttouse4[-components[i, j]-1]    
            #     #             #componentsmedian[i, j] = componenttouse4[-components[i, j]-1]
            #     #             # # 0.81813379 
            #     #             # # 0.77486875
            #     #             # # (?) 

            #     # # print(componentsmedian[infe==gt_masks][components[infe==gt_masks]>=0.].mean())                           
            #     # # print(componentsmedian[infe!=gt_masks][components[infe!=gt_masks]>=0.].mean())    
            #     # # print(np.median(componentsmedian[infe==gt_masks][components[infe==gt_masks]>=0.]))             
            #     # # print(np.median(componentsmedian[infe!=gt_masks][components[infe!=gt_masks]>=0.])) 

            #     # #print(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0].mean()) 
            #     # #print(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0].mean())
            #     # #print(np.median(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0]))
            #     # #print(np.median(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0]))

            #     # # # outputs3touse = np.where(components > 0, outputs3touse, float("nan"))                                                                    
            #     # #componentsmedian = np.where(components > 0, componentsmedian, float("nan"))     

            #     # # #arrayauroc = componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]
            #     # # arrayauroc = componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]
                
            #     # # if (kk == 0) and (vartochange == 0): 
            #     # #     #np.save('arrayforauroc.npy', arrayauroc) 
            #     # #     np.save('arrayforauroc2.npy', arrayauroc)
                
            #     # # else: 
            #     # #     #arrayauroc2 = np.load('arrayforauroc.npy')
            #     # #     arrayauroc2 = np.load('arrayforauroc2.npy')
                    
            #     # #     arrayauroc = np.concatenate((arrayauroc2, arrayauroc), axis=0)  
                    
            #     # #     #np.save('arrayforauroc.npy', arrayauroc)  
            #     # #     np.save('arrayforauroc2.npy', arrayauroc) 

            #     # #formeanvartochange.append(np.nanmean(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))          
            #     # #formeanvartochange.append(np.nanmean(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()])) 
            #     # #formeanvartochange.append(np.nanmedian(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()])) 
            #     # #formeanvartochange.append(np.nanmedian(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanstd(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))         
            #     # #formeanvartochange.append(np.nanstd(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
                
            #     # #formeanvartochange.append(len(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))           
            #     # #formeanvartochange.append(np.count_nonzero(~np.isnan(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))) 
            #     # #formeanvartochange.append(np.count_nonzero(~np.isnan(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))) 
                
            #     # #formeanvartochange.append(np.nanmean(outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))        
            #     # #formeanvartochange.append(np.nanmean(outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()])) 
            #     # #formeanvartochange.append(np.nanmedian(outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanmedian(outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]))
            #     # #formeanvartochange.append(np.nanstd(outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))      
            #     # #formeanvartochange.append(np.nanstd(outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()])) 
            #     # #formeanvartochange.append(np.count_nonzero(~np.isnan(outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]))) 
            #     # #formeanvartochange.append(np.count_nonzero(~np.isnan(outputs3touse[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()])))

            #     # #formeanvartochange.append(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0].mean())    
            #     # #formeanvartochange.append(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0].mean()) 
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0])) 
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>0]))

            #     # # # Median:                       
            #     # # # 0.95357752  
            #     # # # 0.94796602

                
                
                
                
                
                
                
                
                
                
            #     # #sdz

            #     # #asd

            #     # components90at90 = 0.*np.ones_like(components)                                             

            #     # for i in range(components.shape[0]):          
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])                              
            #     #         if components[i, j] >= 0:   
            #     #             #if componenttouse5[components[i, j]-1] >= 0.9:                
            #     #             #if componenttouse5[components[i, j]-1] >= 0.8:  
            #     #             #if componenttouse5[components[i, j]-1] >= 0.85: 
            #     #             # if componenttouse5[components[i, j]-1] >= 0.80:
            #     #             #     components90at90[i, j] = componenttouse5[components[i, j]-1]                      
            #     #             # else:  
            #     #             #     components90at90[i, j] = -componenttouse5[components[i, j]-1] 
            #     #             components90at90[i, j] = componenttouse5[components[i, j]-1]   
                        
            #     #         else:
            #     #             # # (?) 
            #     #             # # (?) 
            #     #             #if componenttouse5[-components[i, j]-1] != 0:
            #     #             #    components90at90[i, j] = -componenttouse5[-components[i, j]-1]
            #     #             #else:
            #     #             #    components90at90[i, j] = -1
            #     #             # # (?)   
            #     #             # # (?) 
                            
            #     #             components90at90[i, j] = -componenttouse5[-components[i, j]-1]   
            #     #             #components90at90[i, j] = componenttouse5[-components[i, j]-1] 

                
                
            #     # #print(components90at90)                                  
            #     # #print(components90at90.shape)    

            #     # # # 8         
            #     # outputs3g = components90at90   
            #     # # # this is per segment (and not per pixel), i.e. 1 value per segment                                 
                
            #     # #adfzs
            #     # fig = plt.figure()                                   
            #     # plt.imshow(render_s2_as_rgb( images[vartochange,:3,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1] ), interpolation="nearest")     
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/input.png'%str(vartochange), bbox_inches='tight')   

            #     # fig = plt.figure()                                     
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                                        
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())     
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())                      
            #     # #plt.imshow(outputs3b[0, :, :].detach().cpu().numpy())      
            #     # #plt.imshow(outputs3c[0, :, :].detach().cpu().numpy())   
            #     # plt.imshow(1.-np.where(outputs3g > 0, outputs3g, 0), cmap='Reds')  
            #     # #plt.imshow(1.-outputs3g, cmap='Reds') 
            #     # plt.colorbar() 
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/greaterthan90.png'%str(vartochange), bbox_inches='tight')    
            #     # #plt.savefig('foldertodownload/greaterthan60.png', bbox_inches='tight')    
            #     # #azsdlf
                
                
                
                
                
            #     # fig = plt.figure()                                      
            #     # #plt.imshow(1.-np.where(componentsmean > 0, componentsmean, 0), cmap='Reds')      
            #     # plt.imshow(1.-componentsmean, cmap='Reds')    
            #     # plt.colorbar() 
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/meanoversegment.png'%str(vartochange), bbox_inches='tight')    
                
            #     # fig = plt.figure()                                       
            #     # #plt.imshow(1.-np.where(componentsmedian > 0, componentsmean, 0), cmap='Reds')           
            #     # plt.imshow(1.-componentsmedian, cmap='Reds')      
            #     # plt.colorbar() 
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/medianoversegment.png'%str(vartochange), bbox_inches='tight')     
                
            #     # #asfasdfas

                

                
                
            #     # # # componententropyoversegment        
            #     # componentsentropy = 0.*np.ones_like(components)                                      
            #     # for i in range(components.shape[0]):          
            #     #     for j in range(components.shape[1]): 
            #     #         if components[i, j] >= 0:   
            #     #             componentsentropy[i, j] = componententropyoversegment[components[i, j]-1]   
            #     #         else:
            #     #             componentsentropy[i, j] = -componententropyoversegment[-components[i, j]-1]  
            #     #             #componentsentropy[i, j] = componententropyoversegment[-components[i, j]-1]

            #     # #print(componentsentropy)         
            #     # #print(componentsentropy.shape)
                
            #     # # # 9 
            #     # outputs3h = componentsentropy
            #     # # # this is per segment (and not per pixel), i.e. 1 value per segment        
                
            #     # #adfzs
            #     # fig = plt.figure()                                     
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                         
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())   
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())          
            #     # #plt.imshow(outputs3b[0, :, :].detach().cpu().numpy())   
            #     # #plt.imshow(outputs3c[0, :, :].detach().cpu().numpy())   
            #     # plt.imshow((100.-np.where(outputs3h > 0, outputs3h, 0))/100., cmap='Reds')  
            #     # #plt.imshow((100.-outputs3h)/100., cmap='Reds') 
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/entroversegm.png'%str(vartochange), bbox_inches='tight')   
            #     # #azsdlf
                
                
                
                
                
            #     # components90at90b = 0.*np.ones_like(components)                                             

            #     # for i in range(components.shape[0]):          
            #     #     for j in range(components.shape[1]): 
            #     #         if components[i, j] >= 0:   
            #     #             components90at90b[i, j] = componenttouse5b[components[i, j]-1]     
                        
            #     #         else:
            #     #             #print(componenttouse5b[-components[i, j]-1])       
            #     #             #if componenttouse5b[-components[i, j]-1] != 0:
            #     #             #    components90at90b[i, j] = -componenttouse5b[-components[i, j]-1] 
            #     #             #else:
            #     #             #    components90at90b[i, j] = -1
                            
            #     #             components90at90b[i, j] = -componenttouse5b[-components[i, j]-1]    
            #     #             #components90at90b[i, j] = componenttouse5b[-components[i, j]-1]

                
                
            #     # # # 10         
            #     # outputs3i = components90at90b
            #     # # # this is per segment (and not per pixel), i.e. 1 value per segment            

            #     # #adfzs
            #     # fig = plt.figure()                                     
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                                 
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())    
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())             
            #     # #plt.imshow(outputs3b[0, :, :].detach().cpu().numpy())   
            #     # #plt.imshow(outputs3c[0, :, :].detach().cpu().numpy())   
            #     # plt.imshow(np.where(outputs3i > 0, outputs3i, 0), cmap='Reds')
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/lessthan60.png'%str(vartochange), bbox_inches='tight')   
            #     # #azsdlf
                
            #     # # #adfzs
            #     # # fig = plt.figure()                                      
            #     # # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                          
            #     # # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())    
            #     # # #plt.imshow(components)   
            #     # # #plt.imshow(components2)   
            #     # # #plt.imshow(outputs3touse)   
            #     # # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())          
            #     # # #plt.imshow(outputs3b[0, :, :].detach().cpu().numpy())   
            #     # # #plt.imshow(outputs3c[0, :, :].detach().cpu().numpy())   
            #     # # #plt.imshow(1.-np.where(outputs3g > 0, outputs3g, 0) + (100.-np.where(outputs3h > 0, outputs3h, 0))/100. + np.where(outputs3i > 0, outputs3i, 0), cmap='Reds')
                
            #     # # def sigmoidfunction(x):
            #     # #   return 1 / (1 + math.exp(-x))
            #     # # sigmoidfunction_v = np.vectorize(sigmoidfunction)
            #     # # #plt.imshow(sigmoidfunction_v(1.-np.where(outputs3g > 0, outputs3g, 0) + (100.-np.where(outputs3h > 0, outputs3h, 0))/100. + np.where(outputs3i > 0, outputs3i, 0)), cmap='Reds') 
            #     # # plt.imshow(1.-sigmoidfunction_v(1.-np.where(outputs3g > 0, outputs3g, 0) + (100.-np.where(outputs3h > 0, outputs3h, 0))/100. + np.where(outputs3i > 0, outputs3i, 0))) 

            #     # # plt.colorbar()
            #     # # plt.axis('off')         
            #     # # plt.savefig('net_input/img18.png', bbox_inches='tight') 
            #     # # #azsdlf

                
                
                
                
            #     # #asdfsadf
            #     # fig = plt.figure()                                      
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                     
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())  
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(1.-outputs3[0, :, :].detach().cpu().numpy(), cmap='Reds')      
            #     # plt.imshow(1.-np.where(components > 0, outputs3[vartochange, :, :].detach().cpu().numpy(), 0), cmap='Reds')        
            #     # #plt.imshow(1.-outputs3[vartochange, :, :].detach().cpu().numpy(), cmap='Reds')      
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/softmaxprob.png'%str(vartochange), bbox_inches='tight')    
            #     # #asdfsadf

            #     # fig = plt.figure()                                      
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                     
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())    
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(1.-outputs3a[0, :, :].detach().cpu().numpy(), cmap='Reds')     
            #     # plt.imshow(1.-np.where(components > 0, outputs3a[vartochange, :, :].detach().cpu().numpy(), 0), cmap='Reds')      
            #     # #plt.imshow(1.-outputs3a[vartochange, :, :].detach().cpu().numpy(), cmap='Reds')      
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/diffDz.png'%str(vartochange), bbox_inches='tight')   
            #     # #asdfasdf

            #     # fig = plt.figure()                                     
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                     
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())    
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())     
            #     # #plt.imshow(-outputs3b[0, :, :].detach().cpu().numpy(), cmap='Reds')    
            #     # #plt.imshow(-outputs3b[vartochange, :, :].detach().cpu().numpy(), cmap='Reds')    
            #     # plt.imshow(-np.where(components > 0, outputs3b[vartochange, :, :].detach().cpu().numpy(), -1), cmap='Reds')     
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/entrperpixel.png'%str(vartochange), bbox_inches='tight')  
            #     # #asdfasdf

            #     # fig = plt.figure()                                     
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                   
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())   
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())        
            #     # #plt.imshow(outputs3b[0, :, :].detach().cpu().numpy())    
            #     # #plt.imshow(1.-outputs3c[0, :, :].detach().cpu().numpy(), cmap='Reds')    
            #     # plt.imshow(1.-np.where(components > 0, outputs3c[vartochange, :, :].detach().cpu().numpy(), 0), cmap='Reds')      
            #     # #plt.imshow(1.-outputs3c[vartochange, :, :].detach().cpu().numpy(), cmap='Reds')      
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/diff1and3.png'%str(vartochange), bbox_inches='tight')   
            #     # #sadfszdf
                
                

            #     # fig = plt.figure()                                     
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                   
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())    
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())        
            #     # #plt.imshow(outputs3b[0, :, :].detach().cpu().numpy())   
            #     # #plt.imshow(outputs3c[0, :, :].detach().cpu().numpy())   
            #     # plt.imshow(-outputs3d[vartochange, :, :].detach().cpu().numpy(), cmap='Reds')   
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/logits1st.png'%str(vartochange), bbox_inches='tight')  
                
            #     # fig = plt.figure()                                     
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                         
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())    
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())          
            #     # #plt.imshow(outputs3b[0, :, :].detach().cpu().numpy())   
            #     # #plt.imshow(outputs3c[0, :, :].detach().cpu().numpy())   
            #     # plt.imshow(-outputs3e[vartochange, :, :].detach().cpu().numpy(), cmap='Reds')   
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/logitsdiffDz.png'%str(vartochange), bbox_inches='tight')   
                
            #     # fig = plt.figure()                                      
            #     # #plt.imshow(labellss[vartochange, :, :].detach().cpu().numpy())                           
            #     # #plt.imshow(theoutput[vartochange, :, :].detach().cpu().numpy())    
            #     # #plt.imshow(components)   
            #     # #plt.imshow(components2)   
            #     # #plt.imshow(outputs3touse)   
            #     # #plt.imshow(outputs3a[0, :, :].detach().cpu().numpy())              
            #     # #plt.imshow(outputs3b[0, :, :].detach().cpu().numpy())   
            #     # #plt.imshow(outputs3c[0, :, :].detach().cpu().numpy())   
            #     # plt.imshow(-outputs3f[vartochange, :, :].detach().cpu().numpy(), cmap='Reds')     
            #     # plt.colorbar()
            #     # plt.axis('off')         
            #     # plt.savefig('foldertodownload%s/logitsdiff1and3.png'%str(vartochange), bbox_inches='tight')

            #     # sadfsa

            
            
            # sadfas



            
            
            
            
            
            
            
            
            
            
            
            
            outputs = outputs.output
            
            
            
            # # regression metrics                                                                                                                                                                                                                        
            error = outputs - labels   
            
            # #squared_error = error**2
            # #test_mse = squared_error.mean().item()
            
            
            
            # #print(test_mse)      
            # #sadfasdf

            # #error = outputs - labels                              
            # #print(error)    
            # #print(error.shape) 
            
            # for iloop in range(error.shape[0]):
            #     for jloop in range(error.shape[2]):
            #         for jjloop in range(error.shape[3]):
            #             #print(torch.from_numpy(np.arange(outputs[iloop,0,jloop,jjloop].item()-outputs2[iloop,0,jloop,jjloop].item(), outputs[iloop,0,jloop,jjloop].item()+outputs2[iloop,0,jloop,jjloop].item()+1.e-06, 1.e-06)).cuda())
            #             #print(labels[iloop,0,jloop,jjloop])
            #             #print(torch.from_numpy(np.arange(outputs[iloop,0,jloop,jjloop].item()-outputs2[iloop,0,jloop,jjloop].item(), outputs[iloop,0,jloop,jjloop].item()+outputs2[iloop,0,jloop,jjloop].item()+1.e-06, 1.e-06)).cuda()-labels[iloop,0,jloop,jjloop])  
            #             #print((torch.from_numpy(np.arange(outputs[iloop,0,jloop,jjloop].item()-outputs2[iloop,0,jloop,jjloop].item(), outputs[iloop,0,jloop,jjloop].item()+outputs2[iloop,0,jloop,jjloop].item()+1.e-06, 1.e-06)).cuda()-labels[iloop,0,jloop,jjloop])[torch.abs(torch.from_numpy(np.arange(outputs[iloop,0,jloop,jjloop].item()-outputs2[iloop,0,jloop,jjloop].item(), outputs[iloop,0,jloop,jjloop].item()+outputs2[iloop,0,jloop,jjloop].item()+1.e-06, 1.e-06)).cuda()-labels[iloop,0,jloop,jjloop]).argmin()])  
            #             #asdfasdf
            
            #             # #error[iloop,0,jloop,jjloop] = torch.min(torch.from_numpy(np.arange(outputs[iloop,0,jloop,jjloop].item()-outputs2[iloop,0,jloop,jjloop].item(), outputs[iloop,0,jloop,jjloop].item()+outputs2[iloop,0,jloop,jjloop].item()+1.e-06, 1.e-06)).cuda()-labels[iloop,0,jloop,jjloop])
            #             # print(iloop)                    
            #             # print(jloop) 
            #             # print(jjloop) 
            #             # #print((torch.from_numpy(np.arange(outputs[iloop,0,jloop,jjloop].item()-outputs2[iloop,0,jloop,jjloop].item(), outputs[iloop,0,jloop,jjloop].item()+outputs2[iloop,0,jloop,jjloop].item()+1.e-06, 1.e-06)).cuda()-labels[iloop,0,jloop,jjloop]))
            #             # print((torch.from_numpy(np.arange(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item()+1.e-06, 1.e-06)).cuda()-labels[iloop,0,jloop,jjloop]))
            #             # #print(labels[iloop,0,jloop,jjloop]) 
            #             # #print(torch.from_numpy(np.arange(outputs[iloop,0,jloop,jjloop].item()-outputs2[iloop,0,jloop,jjloop].item(), outputs[iloop,0,jloop,jjloop].item()+outputs2[iloop,0,jloop,jjloop].item()+1.e-06, 1.e-06)).cuda()) 
            #             # #print((outputs[iloop,0,jloop,jjloop].item()+outputs2[iloop,0,jloop,jjloop].item()-(outputs[iloop,0,jloop,jjloop].item()-outputs2[iloop,0,jloop,jjloop].item()))/1.e-06) 
            #             # #print(round((outputs[iloop,0,jloop,jjloop].item()+outputs2[iloop,0,jloop,jjloop].item()-(outputs[iloop,0,jloop,jjloop].item()-outputs2[iloop,0,jloop,jjloop].item()))/1.e-06))    
            #             # #print(outputs[iloop,0,jloop,jjloop])                                                                 
            #             # #print(outputs2[iloop,0,jloop,jjloop])      
            #             # #try:
            #             # #    print(torch.from_numpy(np.linspace(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), round((outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item()-(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item()))/1.e-06))).cuda()) 
            #             # #except:
            #             # #    print(torch.from_numpy(np.linspace(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), torch.abs(round((outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item()-(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item()))/1.e-06)))).cuda()) 
            #             # print(torch.from_numpy(np.linspace(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), round((outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item()-(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item()))/1.e-06))).cuda()) 
                        
            #             # #asdfzsd                



            #             #print('')   
            #             #print(outputs[iloop,0,jloop,jjloop])
            #             #print(torch.abs(outputs2[iloop,0,jloop,jjloop]))

            #             #error[iloop,0,jloop,jjloop] = (torch.from_numpy(np.arange(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item()+1.e-06, 1.e-06)).cuda()-labels[iloop,0,jloop,jjloop])[torch.abs(torch.from_numpy(np.arange(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item()+1.e-06, 1.e-06)).cuda()-labels[iloop,0,jloop,jjloop]).argmin()]
            #             #error[iloop,0,jloop,jjloop] = (torch.from_numpy(np.linspace(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), round((outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item()-(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item()))/1.e-06))).cuda()-labels[iloop,0,jloop,jjloop])[torch.abs(torch.from_numpy(np.linspace(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), round((outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item()-(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item()))/1.e-06))).cuda()-labels[iloop,0,jloop,jjloop]).argmin()] 
            #             error[iloop,0,jloop,jjloop] = (torch.from_numpy(np.linspace(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), round((outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item()-(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item()))/1.e-08))).cuda()-labels[iloop,0,jloop,jjloop])[torch.abs(torch.from_numpy(np.linspace(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item(), round((outputs[iloop,0,jloop,jjloop].item()+torch.abs(outputs2[iloop,0,jloop,jjloop]).item()-(outputs[iloop,0,jloop,jjloop].item()-torch.abs(outputs2[iloop,0,jloop,jjloop]).item()))/1.e-08))).cuda()-labels[iloop,0,jloop,jjloop]).argmin()]
            
            # # # 0.0024396057706326246
            # # # 0.0016938679618760943  






            # # def pairwise_dist(x, y):    
            # #     xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())                                                             
            # #     rx = (xx.diag().unsqueeze(0).expand_as(xx))          
            # #     ry = (yy.diag().unsqueeze(0).expand_as(yy)) 
            # #     P = (rx.t() + ry - 2*zz)
            # #     return P


            # # def NN_loss(x, y, dim=0): 
            # #     dist = pairwise_dist(x, y)
            # #     values, indices = dist.min(dim=dim)
            # #     return values.mean()

            # #for iloop in range(error.shape[0]):
            # #    for jloop in range(error.shape[2]):
            # #        for jjloop in range(error.shape[3]):
            # #            print(torch.from_numpy(np.arange(outputs[iloop,0,jloop,jjloop].item()-outputs2[iloop,0,jloop,jjloop].item(), outputs[iloop,0,jloop,jjloop].item()+outputs2[iloop,0,jloop,jjloop].item()+1.e-06, 1.e-06)).cuda())
            # #            print(labels[iloop,0,jloop,jjloop])
            # #            #print(torch.from_numpy(np.arange(outputs[iloop,0,jloop,jjloop].item()-outputs2[iloop,0,jloop,jjloop].item(), outputs[iloop,0,jloop,jjloop].item()+outputs2[iloop,0,jloop,jjloop].item()+1.e-06, 1.e-06)).cuda()-labels[iloop,0,jloop,jjloop])  
            # #            print(NN_loss(torch.from_numpy(np.arange(outputs[iloop,0,jloop,jjloop].item()-outputs2[iloop,0,jloop,jjloop].item(), outputs[iloop,0,jloop,jjloop].item()+outputs2[iloop,0,jloop,jjloop].item()+1.e-06, 1.e-06)).cuda(), labels[iloop,0,jloop,jjloop]))
            # #            asdfasdf
            # #
            # #            error[iloop,0,jloop,jjloop] = torch.min(torch.from_numpy(np.arange(outputs[iloop,0,jloop,jjloop].item()-outputs2[iloop,0,jloop,jjloop].item(), outputs[iloop,0,jloop,jjloop].item()+outputs2[iloop,0,jloop,jjloop].item()+1.e-06, 1.e-06)).cuda()-labels[iloop,0,jloop,jjloop])



            # #1.e-06         

            # #for iloop in range(error.shape[0]):
            # #   for jloop in range(error.shape[2]):
            # #       for jjloop in range(error.shape[3]):
            # #           #error[iloop,0,jloop,jjloop] = torch.min((outputs[iloop,0,jloop,jjloop])-labels[iloop,0,jloop,jjloop])
            # #           error[iloop,0,jloop,jjloop] = NN_loss(np.arange(outputs[iloop,0,jloop,jjloop].item()-outputs2[iloop,0,jloop,jjloop].item(), outputs[iloop,0,jloop,jjloop].item()+outputs2[iloop,0,jloop,jjloop].item()+1.e-06, 1.e-06), labels[iloop,0,jloop,jjloop])

            # # def batch_pairwise_dist(a,b): 
            # #     x,y = a,b 
            # #     bs, num_points, points_dim = x.size()
            # #     xx = torch.bmm(x, x.transpose(2,1))
            # #     yy = torch.bmm(y, y.transpose(2,1))
            # #     zz = torch.bmm(x, y.transpose(2,1))
            # #     diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
            # #     rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
            # #     ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
            # #     P = (rx.transpose(2,1) + ry - 2*zz)
            # #     return P


            # # def batch_NN_loss(x, y, dim=1):            
            # #     assert dim != 0   
            # #     pdb.set_trace()
            # #     dist = batch_pairwise_dist(x,y)
            # #     values, indices = dist.min(dim=dim)
            # #     return values.mean(dim=-1)

            # # I have commented 4442 to 4553.    

            # # I have commented 4442 to 4553. 

            squared_error = error**2
            test_mse = squared_error.mean().item()

            
            
            #print(test_mse)                                                                
            #asdfasdf

            
            
            
            
            
            
            
            #asf

            #torch.min()      

            #print(test_mse) 
            #print(torch.sqrt(squared_error.mean()).item())
            #print(outputs2.mean().item()) 
            #sadfasdfs

            test_mae = error.abs().mean().item()
            
            #test_mave = torch.mean(torch.abs(outputs.mean(dim=(1,2)) - labels.mean(dim=(1,2)) ) ).item()
            test_mave = test_mae

            
            
            # print(outputs)                                        
            # print(outputs.shape) 

            # #print(outputs2)
            # print(outputs2.shape)
            # asdfasdf

            
            
            
            
            #print(test_mse)   
            #print(squared_error.median().item())
            #sadfzskd

            # # 0.0016938679618760943 
            # # 6.703515645139699e-14

            


            

            
            
            # regression metrics disguised as classification                                                                                                                                                
            threshold = 0.5       
            label_classification = (labels > threshold).type(torch.int8) 
            output_classification = (outputs > threshold).type(torch.int8)

            diff = output_classification - label_classification   
            fp = torch.count_nonzero(diff==1).item()
            fn = torch.count_nonzero(diff==-1).item()
            tp = label_classification.sum().item() - fn

            test_accuracy = (label_classification==output_classification).type(torch.float).mean().item()
            test_zero_model_mse = (labels**2).mean().item()

            return np.array([test_mse,test_mae,test_mave, test_accuracy,tp,fp,fn,test_zero_model_mse])

    def t_loop(self, epoch, s):  
        
        # self.model.load_state_dict(torch.load('./modelBest18022024c.pt'))   
        # self.model.eval()   
        # self.test()
        
        
        
        
        
        
        
        # sadfasdf
        # import time
        # start_time = time.time()
        # main()
        # print("--- %s seconds ---" % (time.time() - start_time))
        # sadfasdf
        
        
        
        
        
        
        
        
        
        
        # Initialize the running loss                                                                                                                                                                                                                                                                                  
        train_loss = 0.0              
        # # Initialize the progress bar for training 
        train_pbar = tqdm(self.train_loader, total=len(self.train_loader),
                          desc=f"Epoch {epoch + 1}/{self.epochs}") 

        # loop training through batches                 
        for i, (images, labels) in enumerate(train_pbar):  
            # # Move inputs and targets to the device (GPU)                
            images, labels = images.to(self.device), labels.to(self.device)     
            # images.requires_grad = True; labels.requires_grad = True   

            
            
            #print(images.shape)                    
            #print(labels.shape)   

            #asdfasdf



            
            
            
            
            
            # # # (???-)        
            # #self.model.load_state_dict(torch.load('./modelBest18022024c.pt'))      
            # #self.model.eval()
            # # # (???-) 
            # # # 3216 images/ examples   
            
            

            # outputs = self.model(images)       
            # #outputs = outputs.argmax(axis=1).flatten()              
            # #labels = labels.squeeze().flatten() 
            
            
            
            # #print(images.shape)              
            # #print(labels.shape)  
            
            # #asdfzsdf

            # #print(images.shape[0])   
            # # # 3016 +                                             

            
            
            # outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs   

            # # #outputs3 = nn.Softmax(dim=1)(outputsmain)           
            # # outputss33 = nn.Softmax(dim=1)(outputsmain)
            # # #outputs3 = outputs3.max(axis=1)[0] 
            # # outputs3 = outputss33.max(axis=1)[0]

            # # #outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs           

            # # #outputss33 = nn.Softmax(dim=1)(outputsmain)      
            # # #outputs3 = outputss33.max(axis=1)[0]
            
            
            
            
            
            
            
            # theoutput = outputs
            # labellss = labels
            
            # imaini = i     

            # #asdf

            # #asdf
            # def clip_to_quantiles(arr, q_min=0.02, q_max=0.98):   
            #     return np.clip(arr,
            #         np.nanquantile(arr, q_min),
            #         np.nanquantile(arr, q_max),
            #     )    

            # def render_s2_as_rgb(arr):  
            #     # If there are nodata values, lets cast them to zero.                        
            #     if np.ma.isMaskedArray(arr):       
            #         arr = np.ma.getdata(arr.filled(0)) 

            #     # # Select only Blue, green, and red.              
            #     #rgb_slice = arr[:, :, 0:3]  
            #     rgb_slice = arr        

            #     # Clip the data to the quantiles, so the RGB render is not stretched to outliers,
            #     # Which produces dark images.  
            #     for c in [0, 1, 2]:
            #         rgb_slice[:, :, c] = clip_to_quantiles(rgb_slice[:, :, c])

            #     # The current slice is uint16, but we want an uint8 RGB render.       
            #     # We normalise the layer by dividing with the maximum value in the image.
            #     # Then we multiply it by 255 (the max of uint8) to be in the normal RGB range.  
            #     for c in [0, 1, 2]:  
            #         rgb_slice[:, :, c] = (rgb_slice[:, :, c] / rgb_slice[:, :, c].max()) * 255.0

            #     # # We then round to the nearest integer and cast it to uint8.                              
            #     rgb_slice = np.rint(rgb_slice).astype(np.uint8)                 

            #     return rgb_slice         

            # for variabletemporary in range(images.shape[0]):                                
            #     fig = plt.figure()                                   
            #     plt.imshow(render_s2_as_rgb(images[variabletemporary,:3,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]), interpolation="nearest")      
            #     plt.axis('off')           
            #     #plt.savefig('net_input/img{fnam}.png'.format(fnam=1+variabletemporary+3016), bbox_inches='tight')                 
            #     #plt.savefig('net_input/img{fnam}.png'.format(fnam=1+variabletemporary), bbox_inches='tight')             
            #     #plt.savefig('net_input/img{fnam}.png'.format(fnam=1+variabletemporary+(i*images.shape[0])), bbox_inches='tight')         
            #     #plt.savefig('../net_input/img{fnam}.png'.format(fnam=1+variabletemporary+(i*images.shape[0])), bbox_inches='tight')    
            #     #plt.savefig('net_input/img{fnam}.png'.format(fnam=1+variabletemporary+(i*images.shape[0])), bbox_inches='tight')  
            #     #plt.savefig('net_input/img{fnam}.png'.format(fnam=1+variabletemporary+(imaini*images.shape[0])), bbox_inches='tight')  
            #     plt.savefig('../net_input/img{fnam}.png'.format(fnam=1+variabletemporary+(imaini*images.shape[0])), bbox_inches='tight')  
            #     # # imaini
                
            #     #np.save('logits/img{fnam}'.format(fnam=1+variabletemporary+3016), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())       
            #     #np.save('logits/img{fnam}'.format(fnam=1+variabletemporary), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())  
            #     #np.save('logits/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())      
            #     #np.save('../logits/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())  
                
                
                
            #     vartochange = variabletemporary  
                
            #     # import sys        
            #     # sys.path.append('../') 
            #     # sys.path.append('../Automatic-Label-Error-Detection/')
            #     # sys.path.append('../Automatic-Label-Error-Detection/src/')
            #     # #from Automatic-Label-Error-Detection/src/metrics import compute_metrics_components              
            #     # from metrics import compute_metrics_components   
            #     # #print(outputss33.shape)
            #     # #print(labellss.shape)
                
            #     # # fromlabellss = 0.*np.ones_like(outputss33[vartochange,:,:,:].permute(1,2,0).detach().cpu().numpy())       
            #     # # for iuse in range(fromlabellss.shape[0]):   
            #     # #     for juse in range(fromlabellss.shape[1]):
            #     # #         fromlabellss[iuse, juse, labellss[vartochange, iuse, juse].detach().cpu().numpy()] = 1.
                
            #     # #print(fromlabellss)
            #     # #print(fromlabellss.shape)
            #     # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # #_, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 
            #     # #_, components = compute_metrics_components(fromlabellss[:,:,:].copy(), labellss[vartochange,:,:].clone().detach().cpu().numpy()) 

            #     # _, components = compute_metrics_components(outputss33[vartochange,:,:,:].clone().permute(1,2,0).detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy())
                
            #     # #print(outputsmain[vartochange, :, :, :].detach().cpu().numpy())  
            #     # #print(outputsmain[vartochange, :, :, :].detach().cpu().numpy().shape)
                
            #     # componenttouse2 = np.ones_like(range(max(components.max(),-components.min())), dtype=object) 
            #     # #componenttouse2 = np.ones_like(range(components.max()), dtype=object) 
            #     # for i in range(len(componenttouse2)):
            #     #     componenttouse2[i] = [0.,]

            #     # for i in range(components.shape[0]):  
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])          
            #     #         if components[i, j] >= 0:  
            #     #             #components[i, j] = conftouse[components[i, j]-1]                               
            #     #             #tempvariable = components[i, j]             
            #     #             #components[i, j] = conftouse[tempvariable-1] 
            #     #             #components[i, j] = conftouse[0]    
            #     #             #components2[i, j] = conftouse[0]
            #     #             #components2[i, j] = conftouse[components[i, j]-1]
            #     #             if componenttouse2[components[i, j]-1] == [0.,]:
            #     #                 #print(softmax_vs[i, j])
            #     #                 #componenttouse2[components[i, j]-1] = [softmax_vs[i, j],]  
            #     #                 #componenttouse2[components[i, j]-1] = [softmax_vs2[i, j],]  
            #     #                 componenttouse2[components[i, j]-1] = [outputs3[vartochange, i, j].clone().detach().cpu().numpy(),]  
            #     #             else:
            #     #                 #componenttouse2[components[i, j]-1].append(softmax_vs[i, j])      
            #     #                 #componenttouse2[components[i, j]-1].append(softmax_vs2[i, j])  
            #     #                 #componenttouse2[components[i, j]-1].append(outputs3[vartochange, i, j].clone().detach().cpu().numpy()) 
            #     #                 componenttouse2[components[i, j]-1].append(outputs3[vartochange, i, j].clone().detach().cpu().numpy()) 
            #     #             #componenttouse2[components[i, j]-1].append(softmax_vs[i, j]) 
            #     #             #print(conftouse[0]) 
            #     #             #print(components[i, j])               
            #     #             #print(components2[i, j]) 
            #     #         #else: 
            #     #         #    components2[i, j] = -conftouse[-components[i, j]-1]  

            #     # componenttouse3 = []   
            #     # componenttouse4 = []
            #     # componenttouse5 = []
            #     # componenttouse6 = [] 
                
            #     # for componenttouse2a in componenttouse2:
            #     #     #componenttouse3.append(np.mean(componenttouse2a))   
            #     #     componenttouse3.append(np.nanmean(componenttouse2a)) 
            #     #     #componenttouse4.append(np.median(componenttouse2a))
            #     #     componenttouse4.append(np.nanmedian(componenttouse2a))
            #     #     vartoadd = 0   
            #     #     for componenttouse2aa in componenttouse2a: 
            #     #         #if componenttouse2aa >= 0.9:         
            #     #         #if componenttouse2aa >= 0.8: 
            #     #         #if componenttouse2aa >= 0.85:
            #     #         if componenttouse2aa >= 0.80:
            #     #             vartoadd += 1
            #     #     componenttouse5.append(vartoadd)
            #     #     componenttouse6.append(len(componenttouse2a))

            #     # for i in range(len(componenttouse5)):  
            #     #     componenttouse5[i] /= componenttouse6[i]   
            #     # del componenttouse6  

            #     # #print(componenttouse5)   
            #     # #print(np.shape(componenttouse5)) 
            #     # #asdfzsdkf

            #     # #print(componenttouse4)
            #     # #print(np.shape(componenttouse4))
                


            #     # # componentsmean = 0.*np.ones_like(components)          

            #     # # for i in range(components.shape[0]):  
            #     # #     for j in range(components.shape[1]): 
            #     # #         #print(components[i, j])               
            #     # #         if components[i, j] >= 0:  
            #     # #             componentsmean[i, j] = componenttouse3[components[i, j]-1]   
            #     # #         else:
            #     # #             #print(components[i, j])
            #     # #             #print(-components[i, j]-1)
            #     # #             componentsmean[i, j] = -componenttouse3[-components[i, j]-1] 

            #     # # print(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())  
            #     # # print(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())
            #     # # print(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # # print(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # # #sadfsad

            #     # # #formeanvartochange.append(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())   
            #     # # #formeanvartochange.append(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())
            #     # # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # # # # Mean:        
            #     # # # # 0.90298937
            #     # # # # 0.90049725 

                
                
                
                
            #     # componentsmedian = 0.*np.ones_like(components)                  

            #     # for i in range(components.shape[0]):     
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])               
            #     #         if components[i, j] >= 0:  
            #     #             componentsmedian[i, j] = componenttouse4[components[i, j]-1]    
            #     #         else:
            #     #             componentsmedian[i, j] = -componenttouse4[-components[i, j]-1] 

            #     # # print(componentsmedian[infe==gt_masks][components[infe==gt_masks]>=0.].mean())            
            #     # # print(componentsmedian[infe!=gt_masks][components[infe!=gt_masks]>=0.].mean())  
            #     # # print(np.median(componentsmedian[infe==gt_masks][components[infe==gt_masks]>=0.]))   
            #     # # print(np.median(componentsmedian[infe!=gt_masks][components[infe!=gt_masks]>=0.]))

            #     # #print(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean()) 
            #     # #print(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())
            #     # #print(np.median(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # #print(np.median(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # #adfasdf

            #     # #formeanvartochange.append(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean())    
            #     # #formeanvartochange.append(componentsmedian[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.].mean()) 
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() == labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))
            #     # #formeanvartochange.append(np.median(componentsmean[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()][components[theoutput[vartochange,:,:].clone().detach().cpu().numpy() != labellss[vartochange,:,:].clone().detach().cpu().numpy()]>=0.]))

            #     # # # Median:              
            #     # # # 0.95357752  
            #     # # # 0.94796602

                
                
                
                
            #     # components90at90 = 0.*np.ones_like(components)          

            #     # for i in range(components.shape[0]):     
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])                          
            #     #         if components[i, j] >= 0:   
            #     #             #if componenttouse5[components[i, j]-1] >= 0.9:           
            #     #             #if componenttouse5[components[i, j]-1] >= 0.8:  
            #     #             #if componenttouse5[components[i, j]-1] >= 0.85: 
            #     #             if componenttouse5[components[i, j]-1] >= 0.80:
            #     #                 components90at90[i, j] = componenttouse5[components[i, j]-1]   
            #     #             else:
            #     #                 components90at90[i, j] = -componenttouse5[components[i, j]-1]
            #     #         else:
            #     #             components90at90[i, j] = -componenttouse5[-components[i, j]-1]

            #     # componentsmedian = 0.*np.ones_like(components)      

            #     # for i in range(components.shape[0]):       
            #     #     for j in range(components.shape[1]): 
            #     #         #print(components[i, j])                                                                 
            #     #         #if components[i, j] >= 0:
            #     #         if components90at90[i, j] >= 0 and components[i, j] >= 0:  
            #     #             componentsmedian[i, j] = componenttouse4[components[i, j]-1]      
            #     #         elif components[i, j] >= 0:
            #     #             componentsmedian[i, j] = -componenttouse4[components[i, j]-1] 
            #     #         else:
            #     #             componentsmedian[i, j] = -componenttouse4[-components[i, j]-1]
            #     # #sadfasd

            #     # #print(componentsmedian)         
            #     # #print(componentsmedian.shape)  
            #     # #sadfasdf

            #     # #print(vartochange)
            #     # #asdfzsdkf

            #     # fromlabellss = 0.*np.ones_like(outputss33[vartochange,:,:,:].permute(1,2,0).detach().cpu().numpy())     
            #     # for iuse in range(fromlabellss.shape[0]): 
            #     #     for juse in range(fromlabellss.shape[1]):
            #     #         #fromlabellss[iuse, juse, labellss[vartochange, iuse, juse].detach().cpu().numpy()] = 1.       
            #     #         #fromlabellss[iuse, juse, labellss[vartochange, iuse, juse].detach().cpu().numpy()] = componentsmedian[iuse, juse]
            #     #         fromlabellss[iuse, juse, outputs[vartochange, iuse, juse].detach().cpu().numpy()] = componentsmedian[iuse, juse]      

            #     # fromlabellss = np.transpose(fromlabellss, (2,0,1))

            #     # #print(fromlabellss)      
            #     # #print(fromlabellss.shape)

            #     # #print(outputsmain[vartochange, :, :, :].detach().cpu().numpy().shape)  
            #     # #print(fromlabellss.shape)
            #     # #asdfas

            #     # outputsmain2 = np.nanmean( np.array([ outputsmain[vartochange, :, :, :].detach().cpu().numpy(), fromlabellss ]), axis=0 )

            #     # #print(outputsmain2)    
            #     # #print(outputsmain2.shape) 

            #     # #plt.savefig('../net_input/img{fnam}.png'.format(fnam=1+variabletemporary+(i*images.shape[0])), bbox_inches='tight')         

            #     # #np.save('../logits/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())  
            #     # #np.save('../logits/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputsmain2)

            #     # #np.save('logits/img9',  outputsmain[vartochange, :, :, :].detach().cpu().numpy() )           
            #     # #np.save('logits/img9',  outputsmain2 )      
            #     # #np.save('logits/img9{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])),  outputsmain2 )    
            #     # np.save('logits/img{fnam}'.format(fnam=1+variabletemporary+(imaini*images.shape[0])),  outputsmain2 )   
                
                
                
            #     #np.save('logits/img{fnam}'.format(fnam=1+variabletemporary+(imaini*images.shape[0])),  outputsmain[vartochange, :, :, :].detach().cpu().numpy() )   
            #     #np.save('logits/img{fnam}'.format(fnam=1+variabletemporary+(imaini*images.shape[0])),  outputsmain[variabletemporary, :, :, :].detach().cpu().numpy() )  
            #     np.save('../logits/img{fnam}'.format(fnam=1+variabletemporary+(imaini*images.shape[0])),  outputsmain[variabletemporary, :, :, :].detach().cpu().numpy() )  

            #     #np.save('../inference_output/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputs[variabletemporary, :, :].detach().cpu().numpy())   
                
            #     #np.save('../gt_masks/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())
            #     #sadfsadf

            #     #np.save('inference_output/img{fnam}'.format(fnam=1+variabletemporary+3016), outputs[variabletemporary, :, :].detach().cpu().numpy())         
            #     #np.save('inference_output/img{fnam}'.format(fnam=1+variabletemporary), outputs[variabletemporary, :, :].detach().cpu().numpy())    
            #     #np.save('inference_output/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputs[variabletemporary, :, :].detach().cpu().numpy())    
            #     #np.save('../inference_output/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputs[variabletemporary, :, :].detach().cpu().numpy()) 
            #     #np.save('inference_output/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputs[variabletemporary, :, :].detach().cpu().numpy()) 
            #     #np.save('inference_output/img{fnam}'.format(fnam=1+variabletemporary+(imaini*images.shape[0])), outputs[variabletemporary, :, :].detach().cpu().numpy()) 
            #     np.save('../inference_output/img{fnam}'.format(fnam=1+variabletemporary+(imaini*images.shape[0])), outputs[variabletemporary, :, :].detach().cpu().numpy()) 
                
            #     #np.save('gt_masks/img{fnam}'.format(fnam=1+variabletemporary+3016), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())             
            #     #np.save('gt_masks/img{fnam}'.format(fnam=1+variabletemporary), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())     
            #     #np.save('gt_masks/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())   
            #     #np.save('../gt_masks/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())  
            #     #np.save('gt_masks/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())  
            #     #np.save('gt_masks/img{fnam}'.format(fnam=1+variabletemporary+(imaini*images.shape[0])), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())  
            #     np.save('../gt_masks/img{fnam}'.format(fnam=1+variabletemporary+(imaini*images.shape[0])), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())  

            #     plt.close()                                                                                                    
            
            # #sadfasd

            # #if i == 2:                                   
            # #if i == 3: 
            # #if i == 20:
            # #if i == 10:
            # if imaini == 10:
            #     asdfasdf
            # #asdf            
            

            
            
            
            
            
            
            
            
            
            # # self.model.eval()           
            # # with torch.no_grad(): 
            # #     outputs = self.model(images)      
            # #     #outputs = outputs.argmax(axis=1).flatten()            
            # #     #labels = labels.squeeze().flatten() 
                
                
                
            # #     #print(images.shape) 
            # #     #print(labels.shape)
                
            # #     #asdfzsdf

            # #     #print(images.shape[0])   
            # #     # # 3016 +                                       

                
                
            # #     outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs   

            # #     outputs3 = nn.Softmax(dim=1)(outputsmain)      
            # #     outputs3 = outputs3.max(axis=1)[0]
                
            # #     # print(images.shape)          
            # #     # print(labels.shape)  
                
            # #     # print(outputs.shape)    

            # #     # #print(outputs2.shape)           
            # #     # print(outputsmain.shape)   

                
                
            
            #     # def clip_to_quantiles(arr, q_min=0.02, q_max=0.98): 
            #     #     return np.clip(arr,
            #     #         np.nanquantile(arr, q_min),
            #     #         np.nanquantile(arr, q_max),
            #     #     )    

            #     # def render_s2_as_rgb(arr):  
            #     #     # If there are nodata values, lets cast them to zero.               
            #     #     if np.ma.isMaskedArray(arr):       
            #     #         arr = np.ma.getdata(arr.filled(0)) 

            #     #     # # Select only Blue, green, and red.             
            #     #     #rgb_slice = arr[:, :, 0:3]  
            #     #     rgb_slice = arr        

            #     #     # Clip the data to the quantiles, so the RGB render is not stretched to outliers,
            #     #     # Which produces dark images.  
            #     #     for c in [0, 1, 2]:
            #     #         rgb_slice[:, :, c] = clip_to_quantiles(rgb_slice[:, :, c])

            #     #     # The current slice is uint16, but we want an uint8 RGB render.       
            #     #     # We normalise the layer by dividing with the maximum value in the image.
            #     #     # Then we multiply it by 255 (the max of uint8) to be in the normal RGB range.
            #     #     for c in [0, 1, 2]: 
            #     #         rgb_slice[:, :, c] = (rgb_slice[:, :, c] / rgb_slice[:, :, c].max()) * 255.0

            #     #     # # We then round to the nearest integer and cast it to uint8.                      
            #     #     rgb_slice = np.rint(rgb_slice).astype(np.uint8)                

            #     #     return rgb_slice         

            #     # for variabletemporary in range(images.shape[0]):                          
            #     #     fig = plt.figure()                               
            #     #     plt.imshow(render_s2_as_rgb(images[variabletemporary,:3,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]), interpolation="nearest") 
            #     #     plt.axis('off')       
            #     #     #plt.savefig('net_input/img{fnam}.png'.format(fnam=1+variabletemporary+3016), bbox_inches='tight')      
            #     #     #plt.savefig('net_input/img{fnam}.png'.format(fnam=1+variabletemporary), bbox_inches='tight')            
            #     #     #plt.savefig('net_input/img{fnam}.png'.format(fnam=1+variabletemporary+(i*images.shape[0])), bbox_inches='tight')
            #     #     plt.savefig('../net_input/img{fnam}.png'.format(fnam=1+variabletemporary+(i*images.shape[0])), bbox_inches='tight') 

            #     #     #np.save('logits/img{fnam}'.format(fnam=1+variabletemporary+3016), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())   
            #     #     #np.save('logits/img{fnam}'.format(fnam=1+variabletemporary), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())  
            #     #     #np.save('logits/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())
            #     #     np.save('../logits/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())  
                    
            #     #     #np.save('inference_output/img{fnam}'.format(fnam=1+variabletemporary+3016), outputs[variabletemporary, :, :].detach().cpu().numpy()) 
            #     #     #np.save('inference_output/img{fnam}'.format(fnam=1+variabletemporary), outputs[variabletemporary, :, :].detach().cpu().numpy()) 
            #     #     #np.save('inference_output/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputs[variabletemporary, :, :].detach().cpu().numpy())
            #     #     np.save('../inference_output/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputs[variabletemporary, :, :].detach().cpu().numpy()) 
                    
            #     #     #np.save('gt_masks/img{fnam}'.format(fnam=1+variabletemporary+3016), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())   
            #     #     #np.save('gt_masks/img{fnam}'.format(fnam=1+variabletemporary), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())  
            #     #     #np.save('gt_masks/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())
            #     #     np.save('../gt_masks/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())  

            #     #     plt.close()                                                    
                
            #     # #sadfasd

            #     # #if i == 2:    
            #     # if i == 3:
            #     #     asdfasdf
            
            
            
            
            
            # # def clip_to_quantiles(arr, q_min=0.02, q_max=0.98):   
            # #     return np.clip(arr,
            # #         np.nanquantile(arr, q_min),
            # #         np.nanquantile(arr, q_max),
            # #     )  

            # # def render_s2_as_rgb(arr):
            # #     # If there are nodata values, lets cast them to zero.
            # #     if np.ma.isMaskedArray(arr):
            # #         arr = np.ma.getdata(arr.filled(0))

            # #     # # Select only Blue, green, and red.  
            # #     #rgb_slice = arr[:, :, 0:3] 
            # #     rgb_slice = arr   

            # #     # Clip the data to the quantiles, so the RGB render is not stretched to outliers,
            # #     # Which produces dark images. 
            # #     for c in [0, 1, 2]:
            # #         rgb_slice[:, :, c] = clip_to_quantiles(rgb_slice[:, :, c])

            # #     # The current slice is uint16, but we want an uint8 RGB render.  
            # #     # We normalise the layer by dividing with the maximum value in the image.
            # #     # Then we multiply it by 255 (the max of uint8) to be in the normal RGB range.
            # #     for c in [0, 1, 2]:
            # #         rgb_slice[:, :, c] = (rgb_slice[:, :, c] / rgb_slice[:, :, c].max()) * 255.0

            # #     # # We then round to the nearest integer and cast it to uint8.      
            # #     rgb_slice = np.rint(rgb_slice).astype(np.uint8)   

            # #     return rgb_slice   

            
            
            # #print(images.shape)      
            # #print(labels.shape)
            
            
            
            # #outputs = self.model(images)      
            # #outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs   

            # #outputs3 = nn.Softmax(dim=0)(outputs2)     
            
            # #print(outputs.shape)   

            # #print(outputs2.shape)  
            # #print(outputs3.shape)

            
            
            # # #vartemp = 3                        
            # # for vartemp in range(images.shape[0]):         
            # #     fig = plt.figure()             
            # #     plt.imshow(render_s2_as_rgb(images[vartemp,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]), interpolation="nearest")    
            # #     plt.axis('off')    
            # #     plt.savefig('net_input/img{fname}.png'.format(fname=1+(i*vartemp)+vartemp), bbox_inches='tight')  

            # #     np.save('logits/img{fname}'.format(fname=1+(i*vartemp)+vartemp), outputsmain[vartemp, :, :, :].detach().cpu().numpy())   

            # #     np.save('inference_output/img{fname}'.format(fname=1+(i*vartemp)+vartemp), outputs[vartemp, :, :].detach().cpu().numpy()) 

            # #     np.save('gt_masks/img{fname}'.format(fname=1+(i*vartemp)+vartemp), labels[vartemp, :, :, :].detach().cpu().numpy().squeeze())

            # #     plt.close()               

            # # #asdfzsdfs 
            
            
            
            
            
            # #asdfasdf
            # # outputs = self.model(images)      
            # # #outputs = outputs.argmax(axis=1).flatten()            
            # # #labels = labels.squeeze().flatten() 
            
            
            
            # # #print(images.shape) 
            # # #print(labels.shape)
            
            # # #asdfzsdf

            # # #print(images.shape[0])   
            # # # # 3016 +                                       

            
            
            # # outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs   

            # # outputs3 = nn.Softmax(dim=1)(outputsmain)      
            # # outputs3 = outputs3.max(axis=1)[0]
            
            # # # print(images.shape)          
            # # # print(labels.shape)  
            
            # # # print(outputs.shape)    

            # # # #print(outputs2.shape)           
            # # # print(outputsmain.shape)   

            
            
            # # def clip_to_quantiles(arr, q_min=0.02, q_max=0.98): 
            # #     return np.clip(arr,
            # #         np.nanquantile(arr, q_min),
            # #         np.nanquantile(arr, q_max),
            # #     )    

            # # def render_s2_as_rgb(arr):  
            # #     # If there are nodata values, lets cast them to zero.               
            # #     if np.ma.isMaskedArray(arr):       
            # #         arr = np.ma.getdata(arr.filled(0)) 

            # #     # # Select only Blue, green, and red.             
            # #     #rgb_slice = arr[:, :, 0:3]  
            # #     rgb_slice = arr        

            # #     # Clip the data to the quantiles, so the RGB render is not stretched to outliers,
            # #     # Which produces dark images.  
            # #     for c in [0, 1, 2]:
            # #         rgb_slice[:, :, c] = clip_to_quantiles(rgb_slice[:, :, c])

            # #     # The current slice is uint16, but we want an uint8 RGB render.       
            # #     # We normalise the layer by dividing with the maximum value in the image.
            # #     # Then we multiply it by 255 (the max of uint8) to be in the normal RGB range.
            # #     for c in [0, 1, 2]: 
            # #         rgb_slice[:, :, c] = (rgb_slice[:, :, c] / rgb_slice[:, :, c].max()) * 255.0

            # #     # # We then round to the nearest integer and cast it to uint8.                     
            # #     rgb_slice = np.rint(rgb_slice).astype(np.uint8)               

            # #     return rgb_slice        

            # # #if i != 0: 
            # # for variabletemporary in range(images.shape[0]):                         
            # #     fig = plt.figure()                              
            # #     plt.imshow(render_s2_as_rgb(images[variabletemporary,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]), interpolation="nearest") 
            # #     plt.axis('off')       
            # #     #plt.savefig('net_input/img{fnam}.png'.format(fnam=1+variabletemporary+3016), bbox_inches='tight')  
            # #     #plt.savefig('net_input/img{fnam}.png'.format(fnam=1+variabletemporary), bbox_inches='tight')         
            # #     plt.savefig('net_input/img{fnam}.png'.format(fnam=1+variabletemporary+(i*images.shape[0])), bbox_inches='tight')

            # #     #np.save('logits/img{fnam}'.format(fnam=1+variabletemporary+3016), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())  
            # #     #np.save('logits/img{fnam}'.format(fnam=1+variabletemporary), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())  
            # #     np.save('logits/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputsmain[variabletemporary, :, :, :].detach().cpu().numpy())  
                
            # #     #np.save('inference_output/img{fnam}'.format(fnam=1+variabletemporary+3016), outputs[variabletemporary, :, :].detach().cpu().numpy()) 
            # #     #np.save('inference_output/img{fnam}'.format(fnam=1+variabletemporary), outputs[variabletemporary, :, :].detach().cpu().numpy()) 
            # #     np.save('inference_output/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), outputs[variabletemporary, :, :].detach().cpu().numpy()) 
                
            # #     #np.save('gt_masks/img{fnam}'.format(fnam=1+variabletemporary+3016), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())  
            # #     #np.save('gt_masks/img{fnam}'.format(fnam=1+variabletemporary), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())  
            # #     np.save('gt_masks/img{fnam}'.format(fnam=1+variabletemporary+(i*images.shape[0])), labels[variabletemporary, :, :, :].detach().cpu().numpy().squeeze())  

            # #     plt.close()                                          
            
            # # sadfasd

            # # #if i == 2:
            # # #if i == 1:
            # # #    asdfasdf
            
            
            
            # # asdfasdf
            # # for vartemp in range(images.shape[0]):                
            # #     fig = plt.figure()                  
            # #     plt.imshow(render_s2_as_rgb(images[vartemp,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]), interpolation="nearest")       
            # #     plt.axis('off')     
            # #     plt.savefig('net_input/img{fname}.png'.format(fname=1+(i*images.shape[0])+vartemp), bbox_inches='tight')  

            # #     np.save('logits/img{fname}'.format(fname=1+(i*images.shape[0])+vartemp), outputsmain[vartemp, :, :, :].detach().cpu().numpy())   

            # #     np.save('inference_output/img{fname}'.format(fname=1+(i*images.shape[0])+vartemp), outputs[vartemp, :, :].detach().cpu().numpy()) 

            # #     np.save('gt_masks/img{fname}'.format(fname=1+(i*images.shape[0])+vartemp), labels[vartemp, :, :, :].detach().cpu().numpy().squeeze()) 

            # #     plt.close()   
            # # #sadfsadf
            
            # # #outputs = self.model(images)                                
            # # #outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs            
            # # #print(outputs.shape)    
            # # #print(outputs2.shape)
            
            # # #fig = plt.figure()       
            # # #plt.imshow(images[8,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # #plt.axis('off')   
            # # #plt.savefig('net_input/img1.png', bbox_inches='tight')  
            
            # # #np.save('logits/img1', outputsmain[8, :, :, :].detach().cpu().numpy())   
            
            # # #np.save('inference_output/img1', outputs[8, :, :].detach().cpu().numpy())
            
            # # #np.save('gt_masks/img1', labels[8, :, :, :].detach().cpu().numpy().squeeze())
            # # sadfasdf


            
            
            
            
            
            # # # #model.load_state_dict(torch.load('/Data/ndionelis/chaWiTopVieBest4mfr3v.pt'))                       
            # # # # # torch.save(self.model.state_dict(), './modelBest18022024b.pt')                    
            # # # #self.model.load_state_dict(torch.load('./modelBest18022024b.pt'))  
            # # # # # torch.save(self.model.state_dict(), './modelBest18022024c.pt') 
            # # # self.model.load_state_dict(torch.load('./modelBest18022024c.pt'))
            # # self.model.eval() 
            
            
            
            # # #print(images.shape)                      
            # # #print(labels.shape)  
            
            # # outputs = self.model(images)  
            # # outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs

            # # outputs3 = nn.Softmax(dim=0)(outputs2) 
            
            # # print(outputs.shape)  

            # # print(outputs2.shape)
            # # print(outputs3.shape)

            # # #asdfasdfas
            # # fig = plt.figure()      
            # # #ax[0].imshow(np.moveaxis(invimg.numpy(), 0, 2))     
            # # #plt.imshow(np.moveaxis(images[4,:,:,:].permute(1,2,0).detach().cpu().numpy(), 0, 2))    
            # # #plt.imshow(images[4,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()) 
            # # #plt.figure() 
            # # #plt.imshow(s2[...,[3,2,1]]*3e-4)
            # # #plt.imshow(images[4,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy())  
            # # #print((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy().shape)
            # # #asdfasdf
            # # #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]).detach().cpu().numpy()) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # #plt.imshow((images[6,...].permute(1,2,0)[...,[3,2,1]]).detach().cpu().numpy()) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # #plt.imshow((images[8,...].permute(1,2,0)[...,[3,2,1]]).detach().cpu().numpy()) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # #plt.imshow((images[8,...].permute(1,2,0)[...,[3,2,1]]).detach().cpu().numpy()[:, :, [1, 0, 2]]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # #plt.imshow(images[8,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # #plt.imshow(images[6,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # #plt.imshow(images[4,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # #plt.imshow(images[6,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # plt.imshow(images[8,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # plt.axis('off') 
            # # #ax[1].imshow(decoded_mask)       
            # # #ax[1].axis('off')    
            # # #ax[2].imshow(decoded_output)  
            # # #ax[2].imshow(torch.argmax(outputx, 0))              
            # # #ax[2].imshow(torch.max(outputx, 0)[0])   
            # # #ax[2].imshow(probpixelwise)   
            # # #ax[2].axis('off')

            # # #ax[0].set_title('Input Image')                   
            # # #ax[1].set_title('Ground truth')      
            # # #ax[2].set_title('Predicted mask') 

            # # #plt.savefig('result.png', bbox_inches='tight')                             
            # # #plt.savefig('result2.png', bbox_inches='tight')    
            # # #plt.savefig('img1.png', bbox_inches='tight')
            # # #plt.savefig('net_input/img1.png', bbox_inches='tight')                    
            # # #plt.savefig('net_input/img2.png', bbox_inches='tight')   
            # # #plt.savefig('net_input/img3.png', bbox_inches='tight')
            # # #plt.savefig('net_input/img3b.png', bbox_inches='tight')
            # # #plt.savefig('net_input/img3.png', bbox_inches='tight')  
            # # #plt.savefig('net_input/img2.png', bbox_inches='tight') 
            # # #plt.savefig('net_input/img1.png', bbox_inches='tight')
            # # #plt.savefig('net_input/img1b.png', bbox_inches='tight')
            # # #plt.savefig('net_input/img1.png', bbox_inches='tight')
            # # #plt.savefig('net_input/img2.png', bbox_inches='tight')  
            # # #plt.savefig('net_input/img3.png', bbox_inches='tight')  
            # # #plt.savefig('net_input/img2.png', bbox_inches='tight')
            # # plt.savefig('net_input/img1.png', bbox_inches='tight')
            # # #asdfsaf

            # # #asdfasdf



            
            
            # # #print(outputsmain[4, :, :, :].detach().cpu().numpy())
            # # #print(outputsmain[4, :, :, :].detach().cpu().numpy().shape)
            
            # # #print(outputsmain)      
            # # #print(outputsmain.shape)

            # # # [ 1.1124e+00,  1.0282e+00,  5.3334e-01,  ...,  7.6907e-01,
            # # #         -2.7179e-01,  1.9954e-01]]]], device='cuda:0',
            # # #     grad_fn=<ConvolutionBackward0>)   
            # # # torch.Size([32, 11, 128, 128])  

            # # #asdfasd
            # # #outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs      
            # # #outputs3 = nn.Softmax(dim=0)(outputs2)    
            # # # # use: outputsmain[4, :, :, :]
            # # #decoded_output = decode_segmap(torch.argmax(outputx, 0))       
            # # #np.save('logits/img3', outputx.detach().cpu().numpy())  
            # # #np.save('logits/img3', outputsmain[4, :, :, :].detach().cpu().numpy()) 
            # # #np.save('logits/img2', outputsmain[6, :, :, :].detach().cpu().numpy())
            # # #np.save('logits/img1', outputsmain[8, :, :, :].detach().cpu().numpy()) 
            # # #np.save('logits/img2', outputsmain[6, :, :, :].detach().cpu().numpy())
            # # #np.save('logits/img3', outputsmain[4, :, :, :].detach().cpu().numpy()) 
            # # #np.save('logits/img2', outputsmain[6, :, :, :].detach().cpu().numpy())
            # # np.save('logits/img1', outputsmain[8, :, :, :].detach().cpu().numpy())
            
            # # #print(outputsmain[4, :, :, :].detach().cpu().numpy())      
            # # #print(outputsmain[4, :, :, :].detach().cpu().numpy().shape) 

            # # # [ 0.29471564  0.06597131  0.31122494 ...  1.4682894  -0.09499869
            # # #     0.76576567] 
            # # #   [ 0.29224426  0.87665296  0.7934375  ...  0.5582386   0.2070025
            # # #     0.63903254]]]     
            # # # (11, 128, 128)    
            # # #asdfdas

            

            
            
            # # #print(outputs[4, :, :].detach().cpu().numpy())
            # # #print(outputs[4, :, :].detach().cpu().numpy().shape)

            # # #asdfads            
            # # #outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs      
            # # #outputs3 = nn.Softmax(dim=0)(outputs2)     
            # # #decoded_output = decode_segmap(torch.argmax(outputx, 0))    
            # # #np.save('inference_output/img3', torch.argmax(outputx, 0).detach().cpu().numpy())    
            # # #np.save('inference_output/img3', outputs[4, :, :].detach().cpu().numpy())   
            # # #np.save('inference_output/img2', outputs[6, :, :].detach().cpu().numpy())
            # # #np.save('inference_output/img1', outputs[8, :, :].detach().cpu().numpy())  
            # # #np.save('inference_output/img2', outputs[6, :, :].detach().cpu().numpy()) 
            # # #np.save('inference_output/img3', outputs[4, :, :].detach().cpu().numpy())
            # # #np.save('inference_output/img2', outputs[6, :, :].detach().cpu().numpy())
            # # np.save('inference_output/img1', outputs[8, :, :].detach().cpu().numpy())
            # # #asfdasd
            




            # # #print(labels[4, :, :, :].detach().cpu().numpy().squeeze())  
            # # #print(labels[4, :, :, :].detach().cpu().numpy().squeeze().shape)
            
            # # # # use: labels           
            # # # # we use labels[4, :, :, :]   
            # # #decoded_mask = decode_segmap(encode_segmap(seg[sample].clone()))      
            # # #np.save('gt_masks/img3', encode_segmap(seg[sample].clone()).detach().cpu().numpy())         
            # # #np.save('gt_masks/img3', labels[4, :, :, :].detach().cpu().numpy())       
            # # #np.save('gt_masks/img2', labels[6, :, :, :].detach().cpu().numpy())   
            # # #np.save('gt_masks/img1', labels[8, :, :, :].detach().cpu().numpy())     
            # # #np.save('gt_masks/img2', labels[6, :, :, :].detach().cpu().numpy()) 
            # # #np.save('gt_masks/img3', labels[4, :, :, :].detach().cpu().numpy())
            # # #np.save('gt_masks/img3', labels[4, :, :, :].detach().cpu().numpy().squeeze())  
            # # #np.save('gt_masks/img2', labels[6, :, :, :].detach().cpu().numpy().squeeze()) 
            # # np.save('gt_masks/img1', labels[8, :, :, :].detach().cpu().numpy().squeeze())
            # # #sadfasf



            # # fig = plt.figure()       
            # # plt.imshow(labels[8,:,:,:].permute(1,2,0).detach().cpu().numpy()) # plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # plt.axis('off')  
            # # plt.savefig('formeimages/labelsimg1.png', bbox_inches='tight')

            # # fig = plt.figure()       
            # # plt.imshow(outputs.unsqueeze(1)[8,:,:,:].permute(1,2,0).detach().cpu().numpy()) # plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # plt.axis('off')  
            # # plt.savefig('formeimages/predlabimg1.png', bbox_inches='tight')

            # # asdfszdkf

            # # # torch.Size([16, 10, 128, 128])    
            # # # torch.Size([16, 1, 128, 128])  
            # # # torch.Size([16, 128, 128])

            
            
            # # #asdfasdf
            # # # outputs = self.model(images)   
            # # # outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs   

            
            
            # # # #print(images)                 
            # # # print(images.shape) 

            # # # #print(labels)  
            # # # print(labels.shape)

            # # # outputs = self.model(images)  
            # # # #outputs = outputs.argmax(axis=1).flatten()                   
            # # # #labels = labels.squeeze().flatten() 
            
            # # # #outputs = outputs.argmax(axis=1).flatten()                   
            
            # # # #outputs = outputs.argmax(axis=1).flatten()  

            # # # #outputs, outputs2 = outputs.argmax(axis=1).flatten(), outputs.max(axis=1)[0].flatten()         
            # # # #outputs, outputs2 = outputs.argmax(axis=1), outputs.max(axis=1)[0]   
            # # # outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs

            # # # outputs3 = nn.Softmax(dim=0)(outputs2)  



            # # # # #print(outputs)      
            # # # # print(outputs.shape)

            # # # #print(outputs2) 
            # # # print(outputs2.shape)

            # # # #print(outputs3)
            # # # print(outputs3.shape)
            # # # #sadfasdfas



            # # #asdfasdf
            # # #fig, ax = plt.subplots(ncols=1, figsize=(16, 50), facecolor='white')            
            # # # fig = plt.figure()      
            # # # #ax[0].imshow(np.moveaxis(invimg.numpy(), 0, 2))    
            # # # #plt.imshow(np.moveaxis(images[4,:,:,:].permute(1,2,0).detach().cpu().numpy(), 0, 2))   
            # # # #plt.imshow(images[4,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()) 
            # # # #plt.figure()
            # # # #plt.imshow(s2[...,[3,2,1]]*3e-4)
            # # # #plt.imshow(images[4,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy())  
            # # # #print((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy().shape)
            # # # #asdfasdf
            # # # #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]).detach().cpu().numpy()) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # #plt.imshow((images[6,...].permute(1,2,0)[...,[3,2,1]]).detach().cpu().numpy()) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # #plt.imshow((images[8,...].permute(1,2,0)[...,[3,2,1]]).detach().cpu().numpy()) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # #plt.imshow((images[8,...].permute(1,2,0)[...,[3,2,1]]).detach().cpu().numpy()[:, :, [1, 0, 2]]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # #plt.imshow(images[8,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # #plt.imshow(images[6,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # #plt.imshow(images[4,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # #plt.imshow(images[6,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # plt.imshow(images[8,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # plt.axis('off') 
            # # # #ax[1].imshow(decoded_mask)       
            # # # #ax[1].axis('off')    
            # # # #ax[2].imshow(decoded_output)  
            # # # #ax[2].imshow(torch.argmax(outputx, 0))              
            # # # #ax[2].imshow(torch.max(outputx, 0)[0])   
            # # # #ax[2].imshow(probpixelwise)   
            # # # #ax[2].axis('off')

            # # # #ax[0].set_title('Input Image')                   
            # # # #ax[1].set_title('Ground truth')      
            # # # #ax[2].set_title('Predicted mask') 

            # # # #plt.savefig('result.png', bbox_inches='tight')                             
            # # # #plt.savefig('result2.png', bbox_inches='tight')    
            # # # #plt.savefig('img1.png', bbox_inches='tight')
            # # # #plt.savefig('net_input/img1.png', bbox_inches='tight')                    
            # # # #plt.savefig('net_input/img2.png', bbox_inches='tight')   
            # # # #plt.savefig('net_input/img3.png', bbox_inches='tight')
            # # # #plt.savefig('net_input/img3b.png', bbox_inches='tight')
            # # # #plt.savefig('net_input/img3.png', bbox_inches='tight')  
            # # # #plt.savefig('net_input/img2.png', bbox_inches='tight') 
            # # # #plt.savefig('net_input/img1.png', bbox_inches='tight')
            # # # #plt.savefig('net_input/img1b.png', bbox_inches='tight')
            # # # #plt.savefig('net_input/img1.png', bbox_inches='tight')
            # # # #plt.savefig('net_input/img2.png', bbox_inches='tight')  
            # # # #plt.savefig('net_input/img3.png', bbox_inches='tight') 
            # # # #plt.savefig('net_input/img2.png', bbox_inches='tight')
            # # # plt.savefig('net_input/img1.png', bbox_inches='tight')
            # # # #asdfsaf

            # # # #asdfasdf



            
            
            # # # #print(outputsmain[4, :, :, :].detach().cpu().numpy())
            # # # #print(outputsmain[4, :, :, :].detach().cpu().numpy().shape)
            
            # # # #print(outputsmain)      
            # # # #print(outputsmain.shape)

            # # # # [ 1.1124e+00,  1.0282e+00,  5.3334e-01,  ...,  7.6907e-01,
            # # # #         -2.7179e-01,  1.9954e-01]]]], device='cuda:0',
            # # # #     grad_fn=<ConvolutionBackward0>)   
            # # # # torch.Size([32, 11, 128, 128])  

            # # # #asdfasd
            # # # #outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs      
            # # # #outputs3 = nn.Softmax(dim=0)(outputs2)    
            # # # # # use: outputsmain[4, :, :, :]
            # # # #decoded_output = decode_segmap(torch.argmax(outputx, 0))       
            # # # #np.save('logits/img3', outputx.detach().cpu().numpy())  
            # # # #np.save('logits/img3', outputsmain[4, :, :, :].detach().cpu().numpy()) 
            # # # #np.save('logits/img2', outputsmain[6, :, :, :].detach().cpu().numpy())
            # # # #np.save('logits/img1', outputsmain[8, :, :, :].detach().cpu().numpy()) 
            # # # #np.save('logits/img2', outputsmain[6, :, :, :].detach().cpu().numpy())
            # # # #np.save('logits/img3', outputsmain[4, :, :, :].detach().cpu().numpy()) 
            # # # #np.save('logits/img2', outputsmain[6, :, :, :].detach().cpu().numpy())
            # # # np.save('logits/img1', outputsmain[8, :, :, :].detach().cpu().numpy())
            
            # # # #print(outputsmain[4, :, :, :].detach().cpu().numpy())      
            # # # #print(outputsmain[4, :, :, :].detach().cpu().numpy().shape) 

            # # # # [ 0.29471564  0.06597131  0.31122494 ...  1.4682894  -0.09499869
            # # # #     0.76576567] 
            # # # #   [ 0.29224426  0.87665296  0.7934375  ...  0.5582386   0.2070025
            # # # #     0.63903254]]]     
            # # # # (11, 128, 128)    
            # # # #asdfdas

            

            
            
            # # # #print(outputs[4, :, :].detach().cpu().numpy())
            # # # #print(outputs[4, :, :].detach().cpu().numpy().shape)

            # # # #asdfads            
            # # # #outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs      
            # # # #outputs3 = nn.Softmax(dim=0)(outputs2)     
            # # # #decoded_output = decode_segmap(torch.argmax(outputx, 0))    
            # # # #np.save('inference_output/img3', torch.argmax(outputx, 0).detach().cpu().numpy())    
            # # # #np.save('inference_output/img3', outputs[4, :, :].detach().cpu().numpy())   
            # # # #np.save('inference_output/img2', outputs[6, :, :].detach().cpu().numpy())
            # # # #np.save('inference_output/img1', outputs[8, :, :].detach().cpu().numpy())  
            # # # #np.save('inference_output/img2', outputs[6, :, :].detach().cpu().numpy()) 
            # # # #np.save('inference_output/img3', outputs[4, :, :].detach().cpu().numpy())
            # # # #np.save('inference_output/img2', outputs[6, :, :].detach().cpu().numpy())
            # # # np.save('inference_output/img1', outputs[8, :, :].detach().cpu().numpy())
            # # # #asfdasd
            




            # # # #print(labels[4, :, :, :].detach().cpu().numpy().squeeze())  
            # # # #print(labels[4, :, :, :].detach().cpu().numpy().squeeze().shape)
            
            # # # # # use: labels           
            # # # # # we use labels[4, :, :, :]   
            # # # #decoded_mask = decode_segmap(encode_segmap(seg[sample].clone()))      
            # # # #np.save('gt_masks/img3', encode_segmap(seg[sample].clone()).detach().cpu().numpy())         
            # # # #np.save('gt_masks/img3', labels[4, :, :, :].detach().cpu().numpy())       
            # # # #np.save('gt_masks/img2', labels[6, :, :, :].detach().cpu().numpy())   
            # # # #np.save('gt_masks/img1', labels[8, :, :, :].detach().cpu().numpy())     
            # # # #np.save('gt_masks/img2', labels[6, :, :, :].detach().cpu().numpy())
            # # # #np.save('gt_masks/img3', labels[4, :, :, :].detach().cpu().numpy())
            # # # #np.save('gt_masks/img3', labels[4, :, :, :].detach().cpu().numpy().squeeze())  
            # # # #np.save('gt_masks/img2', labels[6, :, :, :].detach().cpu().numpy().squeeze()) 
            # # # np.save('gt_masks/img1', labels[8, :, :, :].detach().cpu().numpy().squeeze()) 
            
            
            
            
            # # #print(images.shape)     
            # # #print(labels.shape)
            # # #sadfasdf

            # # #asdfasdfs



            
            
            
            
            # # # indexiterate = []       
            # # # for iteratemainloop in range(labels.shape[0]):        
            # # #     #print(images[iteratemainloop,:,:,:].shape)                     
            # # #     #print(labels[iteratemainloop,:,:,:].shape) 
                
            # # #     labelsiteration = labels[iteratemainloop,:,:,:]  
            # # #     mask = (labelsiteration==6)     
            # # #     maskedlabels = torch.zeros((labelsiteration.shape))
            # # #     maskedlabels[torch.where(mask)] = 1.0
            # # #     mask = (labelsiteration==7)    
            # # #     maskedlabels[torch.where(mask)] = 1.0
            # # #     mask = (labelsiteration==8)       
            # # #     maskedlabels[torch.where(mask)] = 1.0 
            # # #     mask = (labelsiteration==9)     
            # # #     maskedlabels[torch.where(mask)] = 1.0

            # # #     #print(maskedlabels)  
            # # #     #print(maskedlabels.shape) 
                
            # # #     #print(sum(sum(maskedlabels)).shape) 

            # # #     #print(maskedlabels.sum())   

            # # #     #if maskedlabels.sum() != 0:       
            # # #     if maskedlabels.sum() == 0: 
            # # #         indexiterate.append(iteratemainloop)

            # # #     #print(indexiterate)  
                
            # # # images = images[indexiterate,:,:,:]     
            # # # labels = labels[indexiterate,:,:,:]  
            
            # # # #print(images.shape)             
            # # # #print(labels.shape)              
            


            
            
            
            
            # # #print(labels)
            
            # # #print(images.shape)      
            # # #print(labels.shape)
            
            # # # indexiterate = [] 
            # # # for iteratemainloop in range(labels.shape[0]):   
            # # #     #print(images[iteratemainloop,:,:,:].shape)          
            # # #     #print(labels[iteratemainloop,:,:,:].shape) 
                
            # # #     labelsiteration = labels[iteratemainloop,:,:,:]
            # # #     mask = (labelsiteration==6)    
            # # #     maskedlabels = torch.zeros((labelsiteration.shape))
            # # #     maskedlabels[torch.where(mask)] = 1.0
            # # #     mask = (labelsiteration==7)    
            # # #     maskedlabels[torch.where(mask)] = 1.0
            # # #     mask = (labelsiteration==8)       
            # # #     maskedlabels[torch.where(mask)] = 1.0 
            # # #     mask = (labelsiteration==9)     
            # # #     maskedlabels[torch.where(mask)] = 1.0

            # # #     #print(maskedlabels)  
            # # #     #print(maskedlabels.shape) 
                
            # # #     #print(sum(sum(maskedlabels)).shape) 

            # # #     #print(maskedlabels.sum())   

            # # #     #if maskedlabels.sum() != 0:     
            # # #     if maskedlabels.sum() == 0: 
            # # #         indexiterate.append(iteratemainloop)

            # # #     #print(indexiterate)
                
            # # # images = images[indexiterate,:,:,:]
            # # # labels = labels[indexiterate,:,:,:]
            
            # # # # mask = (labels==6)       
            # # # # maskedlabels = torch.zeros((labels.shape))    
            # # # # maskedlabels[torch.where(mask)] = 1.0 
            # # # # mask = (labels==7)    
            # # # # maskedlabels[torch.where(mask)] = 1.0
            # # # # mask = (labels==8)      
            # # # # maskedlabels[torch.where(mask)] = 1.0 
            # # # # mask = (labels==9)     
            # # # # maskedlabels[torch.where(mask)] = 1.0 

            # # # #print(maskedlabels)   
            # # # #print(maskedlabels.shape)

            # # # #print(sum(maskedlabels)) 
            # # # #print(sum(maskedlabels).shape)

            # # # #print(images.shape)    
            # # # #print(labels.shape)

            
            
            
            
            # # #print(labels)           
            # # #print(labels.shape)

            # # # x = torch.tensor([[1.0, 2.0, 8.0], [-4.0, 0.0, 3.0]])       
            # # # mask = x >=2.0      
            # # # masked_x = torch.zeros((x.shape))  
            # # # masked_x[torch.where(mask)] = 1.0
            # # # print(masked_x)

            # # # mask = labels==0
            # # # maskedlabels = torch.zeros((labels.shape))
            # # # maskedlabels[torch.where(mask)] = 1.0

            # # # #print(maskedlabels)   
            # # # print(maskedlabels.shape) 

            # # # fig = plt.figure()      
            # # # plt.imshow(images[8,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) # plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # plt.axis('off') 
            # # # plt.savefig('formeimages/img1.png', bbox_inches='tight')

            # # # #mask = labels==0       
            # # # mask = labels==7    
            # # # maskedlabels = torch.zeros((labels.shape))
            # # # maskedlabels[torch.where(mask)] = 1.0

            # # # fig = plt.figure()       
            # # # #plt.imshow(labels[8,:,:,:].permute(1,2,0).detach().cpu().numpy()) # plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # plt.imshow(maskedlabels[8,:,:,:].permute(1,2,0).detach().cpu().numpy()) # plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # plt.axis('off')  
            # # # plt.savefig('formeimages/laimg1.png', bbox_inches='tight') 
            
            
            
            
            
            # # #outputs = self.model(images)
            # # #outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs

            
            
            # # # #print(images)              
            # # # print(images.shape) 

            # # # #print(labels)  
            # # # print(labels.shape)

            # # # outputs = self.model(images)  
            # # # #outputs = outputs.argmax(axis=1).flatten()                   
            # # # #labels = labels.squeeze().flatten() 
            
            # # # #outputs = outputs.argmax(axis=1).flatten()                   
            
            # # # #outputs = outputs.argmax(axis=1).flatten()  

            # # # #outputs, outputs2 = outputs.argmax(axis=1).flatten(), outputs.max(axis=1)[0].flatten()         
            # # # #outputs, outputs2 = outputs.argmax(axis=1), outputs.max(axis=1)[0]   
            # # # outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs

            # # # outputs3 = nn.Softmax(dim=0)(outputs2)  



            # # # #print(outputs)      
            # # # print(outputs.shape)

            # # # #print(outputs2) 
            # # # print(outputs2.shape)

            # # # #print(outputs3)
            # # # print(outputs3.shape)
            # # # #sadfasdfas



            # # # #asdfasdf
            # # # #fig, ax = plt.subplots(ncols=1, figsize=(16, 50), facecolor='white')            
            # # # fig = plt.figure()      
            # # # #ax[0].imshow(np.moveaxis(invimg.numpy(), 0, 2))    
            # # # #plt.imshow(np.moveaxis(images[4,:,:,:].permute(1,2,0).detach().cpu().numpy(), 0, 2))   
            # # # #plt.imshow(images[4,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()) 
            # # # #plt.figure()
            # # # #plt.imshow(s2[...,[3,2,1]]*3e-4)
            # # # #plt.imshow(images[4,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy())  
            # # # #print((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy().shape)
            # # # #asdfasdf
            # # # #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]).detach().cpu().numpy()) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # #plt.imshow((images[6,...].permute(1,2,0)[...,[3,2,1]]).detach().cpu().numpy()) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # #plt.imshow((images[8,...].permute(1,2,0)[...,[3,2,1]]).detach().cpu().numpy()) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # #plt.imshow((images[8,...].permute(1,2,0)[...,[3,2,1]]).detach().cpu().numpy()[:, :, [1, 0, 2]]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # #plt.imshow(images[8,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # #plt.imshow(images[6,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # #plt.imshow(images[4,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # #plt.imshow(images[6,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # plt.imshow(images[8,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
            # # # plt.axis('off') 
            # # # #ax[1].imshow(decoded_mask)       
            # # # #ax[1].axis('off')    
            # # # #ax[2].imshow(decoded_output)  
            # # # #ax[2].imshow(torch.argmax(outputx, 0))              
            # # # #ax[2].imshow(torch.max(outputx, 0)[0])   
            # # # #ax[2].imshow(probpixelwise)   
            # # # #ax[2].axis('off')

            # # # #ax[0].set_title('Input Image')                   
            # # # #ax[1].set_title('Ground truth')      
            # # # #ax[2].set_title('Predicted mask') 

            # # # #plt.savefig('result.png', bbox_inches='tight')                             
            # # # #plt.savefig('result2.png', bbox_inches='tight')    
            # # # #plt.savefig('img1.png', bbox_inches='tight')
            # # # #plt.savefig('net_input/img1.png', bbox_inches='tight')                    
            # # # #plt.savefig('net_input/img2.png', bbox_inches='tight')   
            # # # #plt.savefig('net_input/img3.png', bbox_inches='tight')
            # # # #plt.savefig('net_input/img3b.png', bbox_inches='tight')
            # # # #plt.savefig('net_input/img3.png', bbox_inches='tight')  
            # # # #plt.savefig('net_input/img2.png', bbox_inches='tight') 
            # # # #plt.savefig('net_input/img1.png', bbox_inches='tight')
            # # # #plt.savefig('net_input/img1b.png', bbox_inches='tight')
            # # # #plt.savefig('net_input/img1.png', bbox_inches='tight')
            # # # #plt.savefig('net_input/img2.png', bbox_inches='tight')  
            # # # #plt.savefig('net_input/img3.png', bbox_inches='tight') 
            # # # #plt.savefig('net_input/img2.png', bbox_inches='tight')
            # # # plt.savefig('net_input/img1.png', bbox_inches='tight')
            # # # #asdfsaf

            # # # #asdfasdf



            
            
            # # # #print(outputsmain[4, :, :, :].detach().cpu().numpy()) 
            # # # #print(outputsmain[4, :, :, :].detach().cpu().numpy().shape)
            
            # # # #print(outputsmain)      
            # # # #print(outputsmain.shape)

            # # # # [ 1.1124e+00,  1.0282e+00,  5.3334e-01,  ...,  7.6907e-01,
            # # # #         -2.7179e-01,  1.9954e-01]]]], device='cuda:0',
            # # # #     grad_fn=<ConvolutionBackward0>)   
            # # # # torch.Size([32, 11, 128, 128])  

            # # # #asdfasd
            # # # #outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs      
            # # # #outputs3 = nn.Softmax(dim=0)(outputs2)    
            # # # # # use: outputsmain[4, :, :, :]
            # # # #decoded_output = decode_segmap(torch.argmax(outputx, 0))       
            # # # #np.save('logits/img3', outputx.detach().cpu().numpy())  
            # # # #np.save('logits/img3', outputsmain[4, :, :, :].detach().cpu().numpy()) 
            # # # #np.save('logits/img2', outputsmain[6, :, :, :].detach().cpu().numpy())
            # # # #np.save('logits/img1', outputsmain[8, :, :, :].detach().cpu().numpy()) 
            # # # #np.save('logits/img2', outputsmain[6, :, :, :].detach().cpu().numpy())
            # # # #np.save('logits/img3', outputsmain[4, :, :, :].detach().cpu().numpy()) 
            # # # #np.save('logits/img2', outputsmain[6, :, :, :].detach().cpu().numpy())
            # # # np.save('logits/img1', outputsmain[8, :, :, :].detach().cpu().numpy())
            
            # # # #print(outputsmain[4, :, :, :].detach().cpu().numpy())      
            # # # #print(outputsmain[4, :, :, :].detach().cpu().numpy().shape) 

            # # # # [ 0.29471564  0.06597131  0.31122494 ...  1.4682894  -0.09499869
            # # # #     0.76576567] 
            # # # #   [ 0.29224426  0.87665296  0.7934375  ...  0.5582386   0.2070025
            # # # #     0.63903254]]]     
            # # # # (11, 128, 128)    
            # # # #asdfdas

            

            
            
            # # # #print(outputs[4, :, :].detach().cpu().numpy())
            # # # #print(outputs[4, :, :].detach().cpu().numpy().shape)

            # # # #asdfads            
            # # # #outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs      
            # # # #outputs3 = nn.Softmax(dim=0)(outputs2)     
            # # # #decoded_output = decode_segmap(torch.argmax(outputx, 0))    
            # # # #np.save('inference_output/img3', torch.argmax(outputx, 0).detach().cpu().numpy())    
            # # # #np.save('inference_output/img3', outputs[4, :, :].detach().cpu().numpy())   
            # # # #np.save('inference_output/img2', outputs[6, :, :].detach().cpu().numpy())
            # # # #np.save('inference_output/img1', outputs[8, :, :].detach().cpu().numpy())  
            # # # #np.save('inference_output/img2', outputs[6, :, :].detach().cpu().numpy()) 
            # # # #np.save('inference_output/img3', outputs[4, :, :].detach().cpu().numpy())
            # # # #np.save('inference_output/img2', outputs[6, :, :].detach().cpu().numpy())
            # # # np.save('inference_output/img1', outputs[8, :, :].detach().cpu().numpy())
            # # # #asfdasd
            




            # # # #print(labels[4, :, :, :].detach().cpu().numpy().squeeze())  
            # # # #print(labels[4, :, :, :].detach().cpu().numpy().squeeze().shape)
            
            # # # # # use: labels           
            # # # # # we use labels[4, :, :, :]   
            # # # #decoded_mask = decode_segmap(encode_segmap(seg[sample].clone()))      
            # # # #np.save('gt_masks/img3', encode_segmap(seg[sample].clone()).detach().cpu().numpy())         
            # # # #np.save('gt_masks/img3', labels[4, :, :, :].detach().cpu().numpy())       
            # # # #np.save('gt_masks/img2', labels[6, :, :, :].detach().cpu().numpy())   
            # # # #np.save('gt_masks/img1', labels[8, :, :, :].detach().cpu().numpy())     
            # # # #np.save('gt_masks/img2', labels[6, :, :, :].detach().cpu().numpy())
            # # # #np.save('gt_masks/img3', labels[4, :, :, :].detach().cpu().numpy())
            # # # #np.save('gt_masks/img3', labels[4, :, :, :].detach().cpu().numpy().squeeze())  
            # # # #np.save('gt_masks/img2', labels[6, :, :, :].detach().cpu().numpy().squeeze()) 
            # # # np.save('gt_masks/img1', labels[8, :, :, :].detach().cpu().numpy().squeeze()) 
            
            
            
            
            # # #print(images.shape)     
            # # #print(labels.shape)
            
            
            
            
            



            # # # outputs = outputs.flatten()     
            # # # outputs2 = outputs2.flatten()
            
            # # # labels = labels.squeeze().flatten() 
            # # # #labels = labels.flatten()
            
            
            
            
            
            
            
            
            
            
            
            
            
            # # Zero the gradients                                                          
            self.optimizer.zero_grad()         
            
            # get loss            
            #with autocast(dtype=torch.float16):    
            #with autocast(dtype=torch.float16):   
            #with autocast(dtype=torch.float16):
            #with autocast(dtype=torch.float16):
            #with autocast(dtype=torch.float16):
            #with autocast(dtype=torch.float32):
            #with autocast(dtype=torch.float16):
            #with autocast(dtype=torch.float32):
            with autocast(dtype=torch.float16):
                loss = self.get_loss(images, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            train_loss += loss.item()

            
            
            # outputs = outputs.argmax(axis=1).flatten()         
            # #labels = labels.squeeze().flatten() 
            
            

            # #outputs = self.model(images)          
            # #outputs = outputs.argmax(axis=1).flatten()          
            # #labels = labels.squeeze().flatten()
            
            # #outputs = outputs.argmax(axis=1).flatten()    
            
            # #outputs = outputs.argmax(axis=1).flatten()

            # outputs, outputs2 =  outputs.argmax(axis=1).flatten(), outputs.max(axis=1)[0].flatten()

            # print(outputs)
            # print(outputs.shape)
            
            
            
            
            
            
            
            # labels = labels.squeeze().flatten()           
            
            # display progress on console       
            train_pbar.set_postfix({
                "loss": f"{train_loss / (i + 1):.4f}",
                f"lr": self.optimizer.param_groups[0]['lr']})

            # # Update the scheduler    
            if self.lr_scheduler == 'cosine_annealing': 
                s.step() 

        return i, train_loss

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize(x=images, y=labels, y_pred=outputs, images=5,
                            channel_first=True, vmin=0, vmax=1, save_path=f"{self.out_folder}/{name}.png")

    def v_loop(self, epoch):

        # Initialize the progress bar for training  
        val_pbar = tqdm(self.val_loader, total=len(self.val_loader),
                          desc=f"Epoch {epoch + 1}/{self.epochs}") 

        with torch.no_grad(): 
            self.model.eval() 
            val_loss = 0 
            for j, (images, labels) in enumerate(val_pbar):
                # Move inputs and targets to the device (GPU) 
                images, labels = images.to(self.device), labels.to(self.device)

                

                # indexiterate = []   
                # for iteratemainloop in range(labels.shape[0]):    
                #     #print(images[iteratemainloop,:,:,:].shape)                  
                #     #print(labels[iteratemainloop,:,:,:].shape) 
                    
                #     labelsiteration = labels[iteratemainloop,:,:,:]  
                #     mask = (labelsiteration==6)     
                #     maskedlabels = torch.zeros((labelsiteration.shape))
                #     maskedlabels[torch.where(mask)] = 1.0
                #     mask = (labelsiteration==7)    
                #     maskedlabels[torch.where(mask)] = 1.0
                #     mask = (labelsiteration==8)       
                #     maskedlabels[torch.where(mask)] = 1.0 
                #     mask = (labelsiteration==9)     
                #     maskedlabels[torch.where(mask)] = 1.0

                #     #print(maskedlabels)  
                #     #print(maskedlabels.shape) 
                    
                #     #print(sum(sum(maskedlabels)).shape) 

                #     #print(maskedlabels.sum())   

                #     #if maskedlabels.sum() != 0:       
                #     if maskedlabels.sum() == 0: 
                #         indexiterate.append(iteratemainloop)

                #     #print(indexiterate)  
                    
                # images = images[indexiterate,:,:,:]    
                # labels = labels[indexiterate,:,:,:]


                
                # #print(images.shape)                 
                # #print(labels.shape) 
                
                                
                
                
                
                # get loss    
                loss = self.get_loss(images, labels)   
                val_loss += loss.item()

                # # display progress on console                   
                val_pbar.set_postfix({
                    "val_loss": f"{val_loss / (j + 1):.4f}",
                    f"lr": self.optimizer.param_groups[0]['lr']})

            if self.visualise_validation:
                outputs = self.model(images)

                if type(outputs) is tuple:
                    outputs = outputs[0]

                #self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(), outputs.detach().cpu().numpy(), name=f'/val_images/val_{epoch}')

            return j, val_loss

    def save_ckpt(self, epoch, val_loss): 
        #model_sd = self.model.state_dict().copy()    
        #'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()    
        #model_sd = self.model.state_dict().copy()
        #model_sd = self.model.state_dict().copy() 
        model_sd = {'state_dict': self.model.state_dict().copy(), 'optimizer': self.optimizer.state_dict().copy()}  
        
        if self.best_loss is None:
            self.best_epoch = epoch
            self.best_loss = val_loss
            torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_best.pt"))
            self.best_sd = model_sd

        elif self.best_loss > val_loss:
            self.best_epoch = epoch
            self.best_loss = val_loss
            self.epochs_no_improve = 0

            torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_best.pt"))
            self.best_sd = model_sd

        else:
            self.epochs_no_improve += 1

        torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_last.pt"))

    def plot_curves(self, epoch):
        # # visualize loss & lr curves        
        self.e.append(epoch)  

        fig = plt.figure()
        plt.plot(self.e, self.tl, label='Training Loss', )
        plt.plot(self.e, self.vl, label='Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(self.out_folder, f"loss.png"))
        plt.close('all')
        fig = plt.figure()
        plt.plot(self.e, self.lr, label='Learning Rate')
        plt.legend()
        plt.savefig(os.path.join(self.out_folder, f"lr.png"))
        plt.close('all')

    def train(self):
        print("Starting training...")  
        print("")

        # init model    
        self.model.to(self.device) 
        self.model.train()

        # create dst folder for generated files/ artifacts  
        os.makedirs(self.out_folder, exist_ok=True)
        s = self.scheduler

        # # The training loop       
        #for epoch in range(self.epochs):  
        #for epoch in range(1):  
        #for epoch in range(1):  
        #for epoch in range(1):  
        for epoch in range(self.epochs):  
            if epoch == 0 and self.warmup == True:
                s = self.scheduler_warmup
                print('Starting linear warmup phase')
            elif epoch == self.warmup_steps and self.warmup == True:
                s = self.scheduler
                self.warmup = False
                print('Warmup finished')

            i, train_loss = self.t_loop(epoch, s) 
            j, val_loss = self.v_loop(epoch)

            self.tl.append(train_loss / (i + 1))
            self.vl.append(val_loss / (j + 1))
            self.lr.append(self.optimizer.param_groups[0]['lr'])

            # Update the scheduler  
            if self.warmup:
                s.step()
            elif self.lr_scheduler == 'reduce_on_plateau':
                s.step(self.vl[-1])

            # # save check point           
            self.save_ckpt(epoch, val_loss / (j + 1))

            # visualize loss & lr curves   
            #self.plot_curves(epoch)
            self.model.train()

            # Early stopping 
            if self.epochs_no_improve == self.early_stop:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                self.last_epoch = epoch + 1
                break

    def test(self): 
        
        # # Load the best weights                                                         
        # self.model.load_state_dict(self.best_sd)   

        # print("Finished Training. Best epoch: ", self.best_epoch + 1) 
        # print("")     
        # print("Starting Testing...")      
        
        
        
        # #torch.save(self.model.state_dict(), './modelBest18022024.pt')                                                                      
        # #torch.save(self.model.state_dict(), './modelBest18022024b.pt')      
        # torch.save(self.model.state_dict(), './modelBest18022024c.pt')  

        
        


        
        
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi06082024.pt')                        
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi06082024.pt')  
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi06082024.pt')
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi06082024b.pt')
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi06082024c.pt')
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi06082024c.pt')
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi06082024c.pt')
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi06082024c.pt')
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi06082024d.pt')
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi06082024e.pt') # same as d # mse': 0.0038645294083007537, 'mae': 0.014178254413251561, 'mave': 0.005041133719328461, 'acc': 0.9902322825362159, 'precision': 0.6266683857963729, 'recall': 0.3013715439457808, 'baseline_mse': 0.008063177930460196, 'f1': 0.40700839030048186   

        #torch.save(self.model.state_dict(), './modelBestSaveBuildi06082024acopycp.pt')  

        #torch.save(self.model.state_dict(), './modelBestSaveBuildi13082024a.pt')         
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi13082024b.pt')  
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi13082024c.pt')  
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi13082024d.pt')
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi14092024a.pt') 
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi14092024b.pt')
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi14092024c.pt')
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi14092024d.pt')
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi14092024e.pt')
        #torch.save(self.model.state_dict(), './modelBestSaveBuildi14092024f.pt')
        
        torch.save(self.model.state_dict(), './model14012025pt')
        
        

        # torch.save(self.model.state_dict(), './modelBestSaveBuildi13082024e.pt')                                                                                                         
        # torch.save(self.optimizer.state_dict(), './optimodelBestSaveBuildi13082024e.pt') 
        # checkpoint = {
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'optimizer': optimizer.state_dict()
        # }                                                                           
        # save_ckp(checkpoint, is_best, checkpoint_dir, model_dir)                

        #asdfsda

        #sadfas




        
        
        
        
        
        
        
        self.model.eval()    
        test_pbar = tqdm(self.test_loader, total=len(self.test_loader),
                          desc=f"Test Set")            
        
        
        
        #with torch.no_grad():                                                                                                                             
        #with torch.no_grad():       
        #with torch.no_grad():
        #with torch.no_grad():
        #with torch.no_grad():
        if True:
            running_metric = self.get_metrics()     

            for k, (images, labels) in enumerate(test_pbar):  
                images = images.to(self.device)         
                labels = labels.to(self.device) 

                
                
                
                
                
                
                # print(images.shape)                 
                # print(labels.shape)   
                
                # asdfzdklf
                
                
                
                
                
                # outputs = self.model(images)     
                # outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs   

                # outputs3 = nn.Softmax(dim=0)(outputs2)       
                
                # print(outputs.shape)  

                # print(outputs2.shape)
                # print(outputs3.shape)

                # def clip_to_quantiles(arr, q_min=0.02, q_max=0.98):
                #     return np.clip(arr,
                #         np.nanquantile(arr, q_min),
                #         np.nanquantile(arr, q_max),
                #     )    

                # def render_s2_as_rgb(arr):
                #     # If there are nodata values, lets cast them to zero.
                #     if np.ma.isMaskedArray(arr):
                #         arr = np.ma.getdata(arr.filled(0))

                #     # # Select only Blue, green, and red. 
                #     rgb_slice = arr[:, :, 0:3] 

                #     # Clip the data to the quantiles, so the RGB render is not stretched to outliers,
                #     # Which produces dark images. 
                #     for c in [0, 1, 2]:
                #         rgb_slice[:, :, c] = clip_to_quantiles(rgb_slice[:, :, c])

                #     # The current slice is uint16, but we want an uint8 RGB render.  
                #     # We normalise the layer by dividing with the maximum value in the image.
                #     # Then we multiply it by 255 (the max of uint8) to be in the normal RGB range.
                #     for c in [0, 1, 2]:
                #         rgb_slice[:, :, c] = (rgb_slice[:, :, c] / rgb_slice[:, :, c].max()) * 255.0

                #     # # We then round to the nearest integer and cast it to uint8.   
                #     rgb_slice = np.rint(rgb_slice).astype(np.uint8)  

                #     return rgb_slice

                # #asdfasdfas
                # fig = plt.figure()        
                # #ax[0].imshow(np.moveaxis(invimg.numpy(), 0, 2))         
                # #plt.imshow(np.moveaxis(images[4,:,:,:].permute(1,2,0).detach().cpu().numpy(), 0, 2))        
                # #plt.imshow(images[4,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()) 
                # #plt.figure() 
                # #plt.imshow(s2[...,[3,2,1]]*3e-4) 
                # #plt.imshow(images[4,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy())    
                # #print((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy().shape) 
                # #asdfasdf
                # #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]).detach().cpu().numpy()) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
                # #plt.imshow((images[6,...].permute(1,2,0)[...,[3,2,1]]).detach().cpu().numpy()) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
                # #plt.imshow((images[8,...].permute(1,2,0)[...,[3,2,1]]).detach().cpu().numpy()) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
                # #plt.imshow((images[8,...].permute(1,2,0)[...,[3,2,1]]).detach().cpu().numpy()[:, :, [1, 0, 2]]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
                # #plt.imshow(images[8,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
                # #plt.imshow(images[6,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
                # #plt.imshow(render_s2_as_rgb(images[4,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]))
                # plt.imshow(render_s2_as_rgb(images[4,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]), interpolation="nearest") 
                # #plt.imshow(render_s2_as_rgb(images[6,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]), interpolation="nearest")
                # #plt.imshow(render_s2_as_rgb(images[9,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]), interpolation="nearest")
                # #plt.imshow(images[4,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
                # #plt.imshow(images[6,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
                # #plt.imshow(images[9,:vartochange,:,:].permute(1,2,0).detach().cpu().numpy()[:, :, ::-1]) #plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
                # plt.axis('off') 
                # #ax[1].imshow(decoded_mask)         
                # #ax[1].axis('off')      
                # #ax[2].imshow(decoded_output)  
                # #ax[2].imshow(torch.argmax(outputx, 0))               
                # #ax[2].imshow(torch.max(outputx, 0)[0])   
                # #ax[2].imshow(probpixelwise)   
                # #ax[2].axis('off')

                # # # Normalize band DN 
                # # nir_norm = normalize(nir) 
                # # red_norm = normalize(red)
                # # green_norm = normalize(green)

                # # # Stack bands
                # # nrg = np.dstack((nir_norm, red_norm, green_norm))
                
                # #ax[0].set_title('Input Image')                      
                # #ax[1].set_title('Ground truth')       
                # #ax[2].set_title('Predicted mask') 

                # #plt.savefig('result.png', bbox_inches='tight')                                 
                # #plt.savefig('result2.png', bbox_inches='tight')    
                # #plt.savefig('img1.png', bbox_inches='tight')
                # #plt.savefig('net_input/img1.png', bbox_inches='tight')                                
                # #plt.savefig('net_input/img2.png', bbox_inches='tight')    
                # #plt.savefig('net_input/img3.png', bbox_inches='tight')
                # #plt.savefig('net_input/img3b.png', bbox_inches='tight')
                # #plt.savefig('net_input/img3.png', bbox_inches='tight')  
                # #plt.savefig('net_input/img2.png', bbox_inches='tight') 
                # #plt.savefig('net_input/img1.png', bbox_inches='tight')
                # #plt.savefig('net_input/img1b.png', bbox_inches='tight')
                # #plt.savefig('net_input/img1.png', bbox_inches='tight')
                # #plt.savefig('net_input/img2.png', bbox_inches='tight')  
                # plt.savefig('net_input/img3.png', bbox_inches='tight')  
                # #plt.savefig('net_input/img2.png', bbox_inches='tight') 
                # #plt.savefig('net_input/img1.png', bbox_inches='tight')
                # #asdfsaf

                # #asdfasdf



                
                
                # #print(outputsmain[4, :, :, :].detach().cpu().numpy()) 
                # #print(outputsmain[4, :, :, :].detach().cpu().numpy().shape) 
                
                # #print(outputsmain)      
                # #print(outputsmain.shape)

                # # [ 1.1124e+00,  1.0282e+00,  5.3334e-01,  ...,  7.6907e-01,
                # #         -2.7179e-01,  1.9954e-01]]]], device='cuda:0',
                # #     grad_fn=<ConvolutionBackward0>)   
                # # torch.Size([32, 11, 128, 128])  

                # #asdfasd
                # #outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs      
                # #outputs3 = nn.Softmax(dim=0)(outputs2)    
                # # # use: outputsmain[4, :, :, :]
                # #decoded_output = decode_segmap(torch.argmax(outputx, 0))       
                # #np.save('logits/img3', outputx.detach().cpu().numpy())  
                # #np.save('logits/img3', outputsmain[4, :, :, :].detach().cpu().numpy()) 
                # #np.save('logits/img2', outputsmain[6, :, :, :].detach().cpu().numpy())
                # #np.save('logits/img1', outputsmain[8, :, :, :].detach().cpu().numpy()) 
                # #np.save('logits/img2', outputsmain[6, :, :, :].detach().cpu().numpy())
                # np.save('logits/img3', outputsmain[4, :, :, :].detach().cpu().numpy()) 
                # #np.save('logits/img2', outputsmain[6, :, :, :].detach().cpu().numpy())
                # #np.save('logits/img1', outputsmain[9, :, :, :].detach().cpu().numpy())
                
                # #print(outputsmain[4, :, :, :].detach().cpu().numpy())      
                # #print(outputsmain[4, :, :, :].detach().cpu().numpy().shape) 

                # # [ 0.29471564  0.06597131  0.31122494 ...  1.4682894  -0.09499869
                # #     0.76576567] 
                # #   [ 0.29224426  0.87665296  0.7934375  ...  0.5582386   0.2070025
                # #     0.63903254]]]     
                # # (11, 128, 128)    
                # #asdfdas

                

                
                
                # #print(outputs[4, :, :].detach().cpu().numpy())
                # #print(outputs[4, :, :].detach().cpu().numpy().shape)

                # #asdfads            
                # #outputs, outputs2, outputsmain = outputs.argmax(axis=1), outputs.max(axis=1)[0], outputs        
                # #outputs3 = nn.Softmax(dim=0)(outputs2)       
                # #decoded_output = decode_segmap(torch.argmax(outputx, 0))    
                # #np.save('inference_output/img3', torch.argmax(outputx, 0).detach().cpu().numpy())    
                # #np.save('inference_output/img3', outputs[4, :, :].detach().cpu().numpy())   
                # #np.save('inference_output/img2', outputs[6, :, :].detach().cpu().numpy())
                # #np.save('inference_output/img1', outputs[8, :, :].detach().cpu().numpy())  
                # #np.save('inference_output/img2', outputs[6, :, :].detach().cpu().numpy()) 
                # np.save('inference_output/img3', outputs[4, :, :].detach().cpu().numpy())
                # #np.save('inference_output/img2', outputs[6, :, :].detach().cpu().numpy()) 
                # #np.save('inference_output/img1', outputs[9, :, :].detach().cpu().numpy())
                # #asfdasd
                




                # #print(labels[4, :, :, :].detach().cpu().numpy().squeeze())  
                # #print(labels[4, :, :, :].detach().cpu().numpy().squeeze().shape)
                
                # # # use: labels              
                # # # we use labels[4, :, :, :]    
                # #decoded_mask = decode_segmap(encode_segmap(seg[sample].clone()))       
                # #np.save('gt_masks/img3', encode_segmap(seg[sample].clone()).detach().cpu().numpy())          
                # #np.save('gt_masks/img3', labels[4, :, :, :].detach().cpu().numpy())       
                # #np.save('gt_masks/img2', labels[6, :, :, :].detach().cpu().numpy())    
                # #np.save('gt_masks/img1', labels[8, :, :, :].detach().cpu().numpy())     
                # #np.save('gt_masks/img2', labels[6, :, :, :].detach().cpu().numpy()) 
                # #np.save('gt_masks/img3', labels[4, :, :, :].detach().cpu().numpy())
                # np.save('gt_masks/img3', labels[4, :, :, :].detach().cpu().numpy().squeeze())     
                # #np.save('gt_masks/img2', labels[6, :, :, :].detach().cpu().numpy().squeeze())  
                # #np.save('gt_masks/img1', labels[9, :, :, :].detach().cpu().numpy().squeeze())
                # #sadfasf



                # fig = plt.figure()         
                # plt.imshow(labels[4,:,:,:].permute(1,2,0).detach().cpu().numpy()) # plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
                # plt.axis('off')     
                # plt.savefig('formeimages/labelsimg3.png', bbox_inches='tight')     

                # fig = plt.figure()       
                # plt.imshow(outputs.unsqueeze(1)[4,:,:,:].permute(1,2,0).detach().cpu().numpy()) # plt.imshow((images[4,...].permute(1,2,0)[...,[3,2,1]]*3e-4).detach().cpu().numpy())
                # plt.axis('off')        
                # plt.savefig('formeimages/predlabimg3.png', bbox_inches='tight')      

                # asdfszdkf

                
                
                

                #print(k)                
                #sadfasd

                #print(running_metric)
                #print(k)
                #sadfzsdkf

                running_metric += self.get_metrics(images, labels)                                      
                #running_metric += self.get_metrics(images, labels, kk=k)    

                #running_metric /= 2    

                
                
                #print(running_metric)           
                #adsfasdf      

                # # 0.60884081                         

                #if k == 10:           
                #if k == 2: 
                #if k == 5:    
                #if k == 0:
                #if k == 5:
                #    adsfasdf

                #asdfasdfzs

                
                
                
                
                
                
                
                
                # #adfasdf
                # #print(k) 
                # #sadfasd

                
                
                # # if k == 0:
                # #     running_metric = []
                # #     running_metric2 = []

                # #print(running_metric) 
                # #print(k) 
                # #sadfzsdkf

                # #running_metric += self.get_metrics(images, labels)                                    
                # #running_metric += self.get_metrics(images, labels, kk=k)    

                # #running_metricb = self.get_metrics2(images, labels, kk=k)
                # running_metricb, running_metric2b = self.get_metrics2(images, labels, kk=k)

                # # # 0.3677789326270432    
                # # # 0.4258325568070788

                # if k == 0:
                #     running_metric = running_metricb
                #     running_metric2 = running_metric2b

                #     meaniou = 0.

                # elif (k != 1) and ((k-1) % 100 == 0):  
                #     running_metric = running_metricb
                #     running_metric2 = running_metric2b

                # else:
                #     running_metric = np.concatenate((running_metric, running_metricb), axis=0) 
                #     #running_metric.append(running_metricb)
                #     running_metric2 = np.concatenate((running_metric2, running_metric2b), axis=0)

                # #running_metric /= 2        



                # #asda
                # def mean_iou(labels, predictions, n_classes): 
                #     mean_iou = 0.0 
                #     seen_classes = 0

                #     for c in range(n_classes):
                #         labels_c = (labels == c)
                #         pred_c = (predictions == c)

                #         #print(labels_c)
                #         #print(labels_c.shape)
                #         #asdfasdf

                #         labels_c_sum = (labels_c).sum()
                #         pred_c_sum = (pred_c).sum()

                #         #print(labels_c_sum) 
                #         #sadfsadf

                #         if (labels_c_sum > 0) or (pred_c_sum > 0):
                #             seen_classes += 1 

                #             intersect = np.logical_and(labels_c, pred_c).sum()
                #             union = labels_c_sum + pred_c_sum - intersect

                #             #print(intersect / union)  
                #             #print(c)

                #             mean_iou += intersect / union 

                #     #print(seen_classes)     

                #     return mean_iou / seen_classes if seen_classes else 0

                # #meaniou = mean_iou(labellss[vartochange,:,:].clone().detach().cpu().numpy(), theoutput[vartochange,:,:].clone().detach().cpu().numpy(), 11)  
                # #meaniou = mean_iou(labellss[:,:,:].clone().detach().cpu().numpy(), theoutput[:,:,:].clone().detach().cpu().numpy(), 11)

                # #meaniou = mean_iou(running_metric, running_metric2, 11) 
                # # # labellss[:,:,:].clone().detach().cpu().numpy() 
                # # # theoutput[:,:,:].clone().detach().cpu().numpy()

                # #meaniou = mean_iou(labellss[:,:,:].clone().detach().cpu().numpy(), labellss[:,:,:].clone().detach().cpu().numpy(), 11) 
                # #meaniou = mean_iou(np.zeros_like(labellss[:,:,:].clone().detach().cpu().numpy()), labellss[:,:,:].clone().detach().cpu().numpy(), 11)
                # #meaniou = mean_iou(np.ones_like(labellss[:,:,:].clone().detach().cpu().numpy()), labellss[:,:,:].clone().detach().cpu().numpy(), 11)
                # #meaniou = mean_iou(theoutput[:,:,:].clone().detach().cpu().numpy(), theoutput[:,:,:].clone().detach().cpu().numpy(), 11)

                # #print(meaniou)
                # #asda

                # if k == 1446:
                #     meaniou = mean_iou(running_metric, running_metric2, 11)
                #     print(meaniou)

                # if (k != 0) and (k % 100 == 0):
                #     #meaniou = mean_iou(running_metric, running_metric2, 11) 
                #     meaniou += mean_iou(running_metric, running_metric2, 11)
                #     #print(meaniou)
                    
                #     #meaniou /= running_metric.shape[0] 
                #     meaniou /= 100

                #     #running_metric = []  
                #     #running_metric2 = []

                # elif k == 1446:
                #     meaniou += mean_iou(running_metric, running_metric2, 11)
                #     #meaniou /= running_metric.shape[0] 
                #     meaniou /= 46
                #     print(meaniou)
                #     asdfasdf
                
                # # if k == 1446:
                # #     #meaniou = running_metric.mean()    
                # #     meaniou = np.mean(running_metric)
                # #     print(meaniou)
                
                # #print(running_metric)                    
                # #adsfasdf      

                # # # 0.60884081          

                # #if k == 10:    
                # #    adsfasdf

                # #asdfasdfzs

                
                
                # #print(k)                         
                # #sadfzs  



            
            
            
            
            self.test_metrics = self.get_metrics(running_metric=running_metric, k=k)   

            print(f"Test Loss: {self.test_metrics}")                
            #outputs = self.model(images)                   
            #outputs = self.model(images)        
            outputs = self.model(images)        
            #outputs, outputs2 = self.model(images)
            
            #self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(),
            #                   outputs.detach().cpu().numpy(), name='test')          
            
        
        
        if isinstance(self.model, nn.DataParallel):
            #model_sd = self.model.module.state_dict().copy()               
            model_sd = {'state_dict': self.model.module.state_dict().copy(), 'optimizer': self.optimizer.state_dict().copy()}
        else:
            #model_sd = self.model.state_dict().copy()
            model_sd = {'state_dict': self.model.state_dict().copy(), 'optimizer': self.optimizer.state_dict().copy()}
        
        torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_final.pt"))  

    def save_info(self, model_summary=None, n_shot=None, p_split=None, warmup=None, lr=None):
        artifacts = {'training_parameters': {'model': self.name,
                                             'lr': lr,
                                             'scheduler': self.lr_scheduler,
                                             'warm_up': warmup,
                                             'optimizer': str(self.optimizer).split(' (')[0],
                                             'device': str(self.device),
                                             'training_epochs': self.epochs,
                                             'early_stop': self.early_stop,
                                             'train_samples': len(self.train_loader) * model_summary.input_size[0][0],
                                             'val_samples': len(self.val_loader) * model_summary.input_size[0][0],
                                             'test_samples': len(self.test_loader) * model_summary.input_size[0][0],
                                             'n_shot': n_shot,
                                             'p_split': p_split
                                             },

                     'training_info': {'best_val_loss': self.best_loss,
                                       'best_epoch': self.best_epoch,
                                       'last_epoch': self.last_epoch},

                     'test_metrics': self.test_metrics,

                     'plot_info': {'epochs': self.e,
                                   'val_losses': self.vl,
                                   'train_losses': self.tl,
                                   'lr': self.lr},

                     'model_summary': {'batch_size': model_summary.input_size[0],
                                       'input_size': model_summary.total_input,
                                       'total_mult_adds': model_summary.total_mult_adds,
                                       'back_forward_pass_size': model_summary.total_output_bytes,
                                       'param_bytes': model_summary.total_param_bytes,
                                       'trainable_params': model_summary.trainable_params,
                                       'non-trainable_params': model_summary.total_params - model_summary.trainable_params,
                                       'total_params': model_summary.total_params}
                     }

        with open(f"{self.out_folder}/artifacts.json", "w") as outfile:
            json.dump(artifacts, outfile)


class TrainVAE(TrainBase):
    def __init__(self, *args, **kwargs):  # 2048 512     
        super(TrainVAE, self).__init__(*args, **kwargs)
        self.CE_loss = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()
        self.augmentations = transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), value='random'),
                                                 transforms.RandomApply([transforms.RandomResizedCrop(128, scale=(0.8, 1.0),
                                                                                                      ratio=(0.9, 1.1),
                                                                                                      interpolation=2,
                                                                                                      antialias=True),
                                                                         transforms.RandomRotation(degrees=20),
                                                                         transforms.GaussianBlur(kernel_size=3),
                                                                         ], p=0.2),

                                                 # transforms.ColorJitter(
                                                 #     brightness=0.25,
                                                 #     contrast=0.25,
                                                 #     saturation=0.5,
                                                 #     hue=0.05,),
                                                 # transforms.RandomAdjustSharpness(0.5, p=0.2),
                                                 # transforms.RandomAdjustSharpness(1.5, p=0.2),

                                                 ]) 



    def reconstruction_loss(self, reconstruction, original): 
        # Binary Cross-Entropy with Logits Loss   
        batch_size = original.size(0)


        # BCE = F.binary_cross_entropy_with_logits(reconstruction.reshape(batch_size, -1),
        #                                          original.reshape(batch_size, -1), reduction='mean')       

        MSE = F.mse_loss(reconstruction.reshape(batch_size, -1), original.reshape(batch_size, -1), reduction='mean')
        # KLDIV = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  

        return MSE

    def similarity_loss(self, embeddings, embeddings_aug):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_aug = F.normalize(embeddings_aug, p=2, dim=1)
        loss_cos = 1 - F.cosine_similarity(embeddings, embeddings_aug).mean()

        return loss_cos

    def cr_loss(self, mu, logvar, mu_aug, logvar_aug, gamma=1e-3, eps=1e-6):
        std_orig = logvar.exp() + eps
        std_aug = logvar_aug.exp() + eps

        _cr_loss = 0.5 * torch.sum(
            2 * torch.log(std_orig / std_aug) - 1 + (std_aug ** 2 + (mu_aug - mu) ** 2) / std_orig ** 2, dim=1).mean()
        cr_loss = _cr_loss * gamma

        return cr_loss

    def get_loss_aug(self, images, aug_images, labels):

        reconstruction, meta_data, latent = self.model(images)
        reconstruction_aug, meta_data_aug, latent_aug = self.model(aug_images)

        reconstruction_loss = (self.reconstruction_loss(reconstruction=reconstruction, original=images) +
                               self.reconstruction_loss(reconstruction=reconstruction_aug, original=aug_images)) / 2

        kg_labels = labels[:, :31]
        coord_labels = labels[:, 31:34]
        time_labels = labels[:, 34:]
        coord_out, time_out, kg_out = meta_data
        coord_out_aug, time_out_aug, kg_out_aug = meta_data_aug

        kg_loss = (self.CE_loss(kg_out, kg_labels) + self.CE_loss(kg_out_aug, kg_labels)) / 2
        coord_loss = (self.MSE_loss(coord_out, coord_labels) + self.MSE_loss(coord_out_aug, coord_labels)) / 2
        time_loss = (self.MSE_loss(time_out, time_labels) + self.MSE_loss(time_out_aug, time_labels)) / 2

        contrastive_loss = self.similarity_loss(latent, latent_aug)

        loss = reconstruction_loss + kg_loss + coord_loss + time_loss + contrastive_loss
        outputs = (reconstruction, meta_data, latent)

        return loss, reconstruction_loss, kg_loss, coord_loss, time_loss, contrastive_loss, outputs

    def get_loss(self, images, labels):
        reconstruction, meta_data, scale_skip_loss = self.model(images)

        reconstruction_loss = self.reconstruction_loss(reconstruction=reconstruction, original=images)

        kg_labels = labels[:, :31]
        coord_labels = labels[:, 31:34]
        time_labels = labels[:, 34:]
        coord_out, time_out, kg_out = meta_data

        kg_loss = self.CE_loss(kg_out, kg_labels)
        coord_loss = self.MSE_loss(coord_out, coord_labels)
        time_loss = self.MSE_loss(time_out, time_labels)

        # loss = 0.5*reconstruction_loss + 0.25*kg_loss + 0.125*coord_loss + 0.125*time_loss + scale_skip_loss
        loss = reconstruction_loss + kg_loss + coord_loss + time_loss + scale_skip_loss
        outputs = (reconstruction, meta_data, scale_skip_loss)

        return loss, reconstruction_loss, kg_loss, coord_loss, time_loss, scale_skip_loss, outputs

    def t_loop(self, epoch, s):
        # Initialize the running loss 
        train_loss = 0.0 
        train_reconstruction_loss = 0.0
        train_kg_loss = 0.0
        train_coord_loss = 0.0
        train_time_loss = 0.0
        train_scale_skip_loss = 0.0

        # Initialize the progress bar for training
        train_pbar = tqdm(self.train_loader, total=len(self.train_loader),
                          desc=f"Epoch {epoch + 1}/{self.epochs}")

        # loop training through batches
        for i, (images, labels) in enumerate(train_pbar):
            # Move inputs and targets to the device (GPU)
            images, labels = images.to(self.device), labels.to(self.device)


            # Zero the gradients 
            self.optimizer.zero_grad()
            # get loss
            with autocast(dtype=torch.float16):
                loss, reconstruction_loss, kg_loss, coord_loss, time_loss, scale_skip_loss, outputs = self.get_loss(images, labels)
                # loss, outputs = self.get_loss(images, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            train_loss += loss.item()
            train_kg_loss += kg_loss.item()
            train_coord_loss += coord_loss.item()
            train_time_loss += time_loss.item()
            train_reconstruction_loss += reconstruction_loss.item()
            train_scale_skip_loss += scale_skip_loss.item()

            # display progress on console
            train_pbar.set_postfix({
                "loss": f"{train_loss / (i + 1):.4f}",
                "loss_kg": f"{train_kg_loss / (i + 1):.4f}",
                "loss_coord": f"{train_coord_loss / (i + 1):.4f}",
                "loss_time": f"{train_time_loss / (i + 1):.4f}",
                "loss_reconstruction": f"{train_reconstruction_loss / (i + 1):.4f}",
                "scale_skip_loss": f"{train_scale_skip_loss / (i + 1):.4f}",
                f"lr": self.optimizer.param_groups[0]['lr']})

            # # Update the scheduler
            if self.lr_scheduler == 'cosine_annealing':
                s.step()

            if (i % 10000) == 0 and i != 0:
                self.val_visualize(images, labels, outputs, name=f'/val_images/train_{epoch}_{i}')
                model_sd = self.model.state_dict()
                torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_ckpt.pt"))

        return i, train_loss

    def v_loop(self, epoch):

        # Initialize the progress bar for training
        val_pbar = tqdm(self.val_loader, total=len(self.val_loader),
                          desc=f"Epoch {epoch + 1}/{self.epochs}")

        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            val_reconstruction_loss = 0.0
            val_kg_loss = 0.0
            val_coord_loss = 0.0
            val_time_loss = 0.0
            val_scale_skip_loss = 0.0

            for j, (images, labels) in enumerate(val_pbar):
                # Move inputs and targets to the device (GPU)
                images, labels = images.to(self.device), labels.to(self.device)

                # get loss
                loss, reconstruction_loss, kg_loss, coord_loss, time_loss, scale_skip_loss, outputs = self.get_loss(images, labels)

                val_loss += loss.item()
                val_kg_loss += kg_loss.item()
                val_coord_loss += coord_loss.item()
                val_time_loss += time_loss.item()
                val_reconstruction_loss += reconstruction_loss.item()
                val_scale_skip_loss += scale_skip_loss.item()

                # display progress on console
                val_pbar.set_postfix({
                    "val_loss": f"{val_loss / (j + 1):.4f}",
                    "loss_kg": f"{val_kg_loss / (j + 1):.4f}",
                    "loss_coord": f"{val_coord_loss / (j + 1):.4f}",
                    "loss_time": f"{val_time_loss / (j + 1):.4f}",
                    "loss_reconstruction": f"{val_reconstruction_loss / (j + 1):.4f}",
                    "scale_skip_loss": f"{val_scale_skip_loss / (j + 1):.4f}",
                    f"lr": self.optimizer.param_groups[0]['lr']})

            if self.visualise_validation:
                self.val_visualize(images, labels, outputs, name=f'/val_images/val_{epoch}')

            return j, val_loss

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_vae(images=images, labels=labels, outputs=outputs, num_images=5, channel_first=True,
                                save_path=f"{self.out_folder}/{name}.png")

class TrainLandCover(TrainBase):  

    def set_criterion(self): 
        return nn.CrossEntropyLoss()  

    def get_loss(self, images, labels): 
        #outputs = self.model(images)                      
        #outputs = self.model(images)  
        #outputs = self.model(images)
        #outputs = self.model(images)
        #outputs = self.model(images)
        
        
        
        #print(images)  
        #print(images.shape)
        
        # # torch.Size([128, 10, 128, 128])                    
        
        #images = images[:, :3, :, :]     
        
        #asdfsa   
        
        
        
        outputs = self.model(images)            
        
        
        
        outputs = outputs.output
        
        
        
        
        
        
        #print(outputs)               
        #print(outputs.shape) 
        #sdfsadf
        
        # # sadfzsdf
        
        
        
        
        
        
        
        
        
        
        #print(outputs)      
        #print(outputs.shape)
        #sdfsadf
        
        # # torch.Size([128, 512, 4, 4])     
        
        
        
        # print(outputs.shape) 
        # print(labels.shape)
        
        # sadfasdf
        
        # # # torch.Size([1, 11, 116, 116])
        # # # torch.Size([32, 1, 128, 128])
        
        
        
        
        
        
        
        
        
        #outputs = self.model(images, images, images, images, images) # # (img, ori_img, top_start, left_start, crop_size)        
        outputs = outputs.flatten(start_dim=2).squeeze() 
        labels = labels.flatten(start_dim=1).squeeze()
        loss = self.criterion(outputs, labels)
        #loss += smp.losses.DiceLoss(mode="multiclass", classes=11)(y_pred=outputs, y_true=labels)       
        #loss += smp.losses.DiceLoss(mode="multiclass", classes=11)(y_pred=outputs, y_true=labels) 
        #loss += smp.losses.DiceLoss(mode="multiclass", classes=11)(y_pred=outputs, y_true=labels)
        
        return loss

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_lc(x=images, y=labels, y_pred=outputs.argmax(axis=1), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")

    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        
        if (running_metric is not None) and (k is not None):
            metric_names = ['acc','precision','recall','baseline_mse']
            # intermediary_values = ['confusion_matrix']  

            confmat = running_metric

            total_pixels = np.sum(confmat)
            
            tp_per_class = np.diagonal(confmat)
            total_tp = tp_per_class.sum()

            fp_per_class = confmat.sum(axis=0) - tp_per_class
            fn_per_class = confmat.sum(axis=1) - tp_per_class

            precision_per_class = tp_per_class/(fp_per_class+tp_per_class)
            recall_per_class = tp_per_class/(fn_per_class+tp_per_class)

            precision_micro = total_tp/(fp_per_class.sum() + total_tp)
            recall_micro = total_tp/(fn_per_class.sum() + total_tp)
            precision_macro = np.mean(precision_per_class)
            recall_macro = np.mean(recall_per_class)

            acc_total = total_tp/total_pixels

            final_metrics = {'acc':acc_total, 'precision_per_class':precision_per_class.tolist(),'recall_per_class':recall_per_class.tolist() ,'precision_micro':precision_micro, 'precision_macro':precision_macro, 'recall_micro':recall_micro, 'recall_macro':recall_macro, 'conf_mat':confmat.tolist()}

            return final_metrics


        elif (images == None) and (labels == None):
            intermediary_values = ['confusion_matrix']
            num_classes = len(config_lc.lc_raw_classes.keys())
            metric_init = np.zeros((num_classes,num_classes)) # 
            return  metric_init
        
        
        else:
            outputs = self.model(images)
            
            
            
            outputs = outputs.output 
            
            
            
            
            
            
            
            
            outputs = outputs.argmax(axis=1).flatten()
            labels = labels.squeeze().flatten()
            
            # # from pytorch confusion matrix  
            num_classes = len(config_lc.lc_raw_classes.keys())
            unique_mapping = labels.to(torch.long) * num_classes + outputs.to(torch.long)
            bins = torch.bincount(unique_mapping, minlength=num_classes**2) 
            cfm = bins.reshape(num_classes, num_classes)

            return cfm.cpu().numpy()









class TrainClassificationBuildings(TrainBase): 

    def set_criterion(self):
        return nn.CrossEntropyLoss(weight=torch.tensor([2.65209613e-01, 6.95524031e-01,
                                                        3.12650858e-02, 7.95257252e-03, 4.86978615e-05]))

    def get_loss(self, images, labels):
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_lc_classification(x=images, y=labels, y_pred=outputs, images=5,
                                              channel_first=True, num_classes=5,
                                              labels=['no urbanization', 'sparse urbanization',
                                                      'moderate urbanization', 'significant urbanization',
                                                      'extreme urbanization'],
                                              save_path=f"{self.out_folder}/{name}.png")

    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):

        if (running_metric is not None) and (k is not None):
            metric_names = ['mse','mae','mave','acc','precision','recall','baseline_mse']
            # intermediary_values = ['mse','mae','mave','acc','tp','fp','fn','baseline_mse']

            final_metrics = {'mse':running_metric[0] / (k + 1), 'mae':running_metric[1] / (k + 1), 'acc':running_metric[2]/ (k + 1)}

            return final_metrics

        elif (images == None) and (labels == None):
            intermediary_values = ['mse','mae','acc']
            metric_init = np.zeros(len(intermediary_values)) #
            return  metric_init

        else:
            outputs = self.model(images)

            # regression metrics
            error = outputs - labels
            squared_error = error ** 2
            test_mse = squared_error.mean().item()
            test_mae = error.abs().mean().item()
            # test_mave = torch.mean(torch.abs(outputs.mean(dim=(1, 2)) - labels.mean(dim=(1, 2)))).item()

            # regression metrics disguised as classification
            output_classification = outputs.argmax(axis=1).flatten()
            label_classification = labels.argmax(axis=1).flatten()

            test_accuracy = (label_classification == output_classification).type(torch.float).mean().item()

            return np.array([test_mse, test_mae, test_accuracy])




class TrainClassificationLC(TrainClassificationBuildings):

    def set_criterion(self):
        return nn.CrossEntropyLoss()
    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_lc_classification(x=images, y=labels, y_pred=outputs, images=5,
                                              channel_first=True, num_classes=11,
                                              labels=['Tree cover', 'Shrubland', 'Grassland', 'Cropland', 'Built-up',
                                                      'Bare/sparse', 'snow/ice','Perm water', 'Wetland', 'Mangroves',
                                                      'Moss'],
                                              save_path=f"{self.out_folder}/{name}.png")


class TrainClassificationRoads(TrainClassificationBuildings):

    def set_criterion(self):
        return nn.CrossEntropyLoss(weight=torch.tensor([0.37228453, 0.62771547]))

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_lc_classification(x=images, y=labels, y_pred=outputs, images=5,
                                              channel_first=True, num_classes=2,
                                              labels=['No Roads', 'Roads'],
                                              save_path=f"{self.out_folder}/{name}.png")


class TrainViT(TrainBase):
    def get_loss(self, images, labels):
        outputs = self.model(images)
        labels = self.model.patchify(labels)
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        outputs = self.model.unpatchify(torch.from_numpy(outputs), c=labels.shape[1])
        visualize.visualize(x=images, y=labels, y_pred=outputs.detach().cpu().numpy(), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")

    def v_loop(self, epoch):

        # Initialize the progress bar for training
        val_pbar = tqdm(self.val_loader, total=len(self.val_loader),
                          desc=f"Epoch {epoch + 1}/{self.epochs}")

        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            for j, (images, labels) in enumerate(val_pbar):
                # Move inputs and targets to the device (GPU)
                images, labels = images.to(self.device), labels.to(self.device)

                # get loss
                loss = self.get_loss(images, labels)
                val_loss += loss.item()

                # display progress on console
                val_pbar.set_postfix({
                    "val_loss": f"{val_loss / (j + 1):.4f}",
                    f"lr": self.optimizer.param_groups[0]['lr']})

            if self.visualise_validation:
                outputs = self.model(images[:, :, 16:-16, 16:-16])

                if type(outputs) is tuple:
                    outputs = outputs[0]

                self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(), outputs.detach().cpu().numpy(), name=f'/val_images/val_{epoch}')

            return j, val_loss


class TrainSatMAE(TrainBase):
    def get_loss(self, images, labels):
        images = images[:, :, 16:-16, 16:-16]
        labels = labels[:, :, 16:-16, 16:-16]
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        images = images[:, :, 16:-16, 16:-16]
        labels = labels[:, :, 16:-16, 16:-16]
        visualize.visualize(x=images, y=labels, y_pred=outputs.detach().cpu().numpy(), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")

    def v_loop(self, epoch):

        # Initialize the progress bar for training
        val_pbar = tqdm(self.val_loader, total=len(self.val_loader),
                          desc=f"Epoch {epoch + 1}/{self.epochs}")

        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            for j, (images, labels) in enumerate(val_pbar):
                # Move inputs and targets to the device (GPU)
                images, labels = images.to(self.device), labels.to(self.device)

                # get loss
                loss = self.get_loss(images, labels)
                val_loss += loss.item()

                # display progress on console
                val_pbar.set_postfix({
                    "val_loss": f"{val_loss / (j + 1):.4f}",
                    f"lr": self.optimizer.param_groups[0]['lr']})

            if self.visualise_validation:
                outputs = self.model(images[:, :, 16:-16, 16:-16])

                if type(outputs) is tuple:
                    outputs = outputs[0]

                self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(), outputs.detach().cpu().numpy(), name=f'/val_images/val_{epoch}')

            return j, val_loss


class TrainSatMAE_lc(TrainLandCover):
    def get_loss(self, images, labels):
        images = images[:, :, 16:-16, 16:-16]
        labels = labels[:, :, 16:-16, 16:-16]
        outputs = self.model(images)
        outputs = outputs.flatten(start_dim=2).squeeze()
        labels = labels.flatten(start_dim=1).squeeze()
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        images = images[:, :, 16:-16, 16:-16]
        labels = labels[:, :, 16:-16, 16:-16]
        visualize.visualize_lc(x=images, y=labels, y_pred=outputs.argmax(axis=1), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")

    def v_loop(self, epoch):

        # Initialize the progress bar for training
        val_pbar = tqdm(self.val_loader, total=len(self.val_loader),
                          desc=f"Epoch {epoch + 1}/{self.epochs}")

        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            for j, (images, labels) in enumerate(val_pbar):
                # Move inputs and targets to the device (GPU)
                images, labels = images.to(self.device), labels.to(self.device)

                # get loss
                loss = self.get_loss(images, labels)
                val_loss += loss.item()

                # display progress on console
                val_pbar.set_postfix({
                    "val_loss": f"{val_loss / (j + 1):.4f}",
                    f"lr": self.optimizer.param_groups[0]['lr']})

            if self.visualise_validation:
                outputs = self.model(images[:, :, 16:-16, 16:-16])

                if type(outputs) is tuple:
                    outputs = outputs[0]

                self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(), outputs.detach().cpu().numpy(), name=f'/val_images/val_{epoch}')

            return j, val_loss

    def test(self):
        # Load the best weights
        self.model.load_state_dict(self.best_sd)

        print("Finished Training. Best epoch: ", self.best_epoch + 1)
        print("")
        print("Starting Testing...")
        self.model.eval()
        test_pbar = tqdm(self.test_loader, total=len(self.test_loader),
                         desc=f"Test Set")
        with torch.no_grad():
            running_metric = self.get_metrics()

            for k, (images, labels) in enumerate(test_pbar):
                images = images[:, :, 16:-16, 16:-16].to(self.device)
                labels = labels[:, :, 16:-16, 16:-16].to(self.device)

                running_metric += self.get_metrics(images, labels)

            self.test_metrics = self.get_metrics(running_metric=running_metric, k=k)

            print(f"Test Loss: {self.test_metrics}")
            outputs = self.model(images)
            self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(),
                               outputs.detach().cpu().numpy(), name='test')


class TrainViTLandCover(TrainBase):

    def set_criterion(self):
        return nn.CrossEntropyLoss()

    def get_loss(self, images, labels):
        outputs = self.model.unpatchify(self.model(images), c=11).flatten(start_dim=2).squeeze()
        labels = labels.flatten(start_dim=1).squeeze()
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        outputs = self.model.unpatchify(torch.from_numpy(outputs), c=11)
        visualize.visualize_lc(x=images, y=labels, y_pred=outputs.detach().cpu().numpy().argmax(axis=1), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")

    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        
        if (running_metric is not None) and (k is not None):
            metric_names = ['acc','precision','recall','baseline_mse']
            # intermediary_values = ['confusion_matrix']

            confmat = running_metric

            total_pixels = np.sum(confmat)
            
            tp_per_class = np.diagonal(confmat)
            total_tp = tp_per_class.sum()

            fp_per_class = confmat.sum(axis=0) - tp_per_class
            fn_per_class = confmat.sum(axis=1) - tp_per_class
            

            precision_per_class = tp_per_class/(fp_per_class+tp_per_class)
            recall_per_class = tp_per_class/(fn_per_class+tp_per_class)

            precision_micro = total_tp/(fp_per_class.sum() + total_tp)
            recall_micro = total_tp/(fn_per_class.sum() + total_tp)
            precision_macro = np.mean(precision_per_class)
            recall_macro = np.mean(recall_per_class)

            acc_total = total_tp/total_pixels

            final_metrics = {'acc':acc_total, 'precision_per_class':precision_per_class.tolist(),'recall_per_class':recall_per_class.tolist() ,'precision_micro':precision_micro, 'precision_macro':precision_macro, 'recall_micro':recall_micro, 'recall_macro':recall_macro, 'conf_mat':confmat.tolist()}

            return final_metrics


        elif (images == None) and (labels == None):
            intermediary_values = ['confusion_matrix']
            num_classes = len(config_lc.lc_raw_classes.keys())
            metric_init = np.zeros((num_classes,num_classes)) # 
            return  metric_init
        
        
        else:
            outputs = self.model.unpatchify(self.model(images), c=11)
            outputs = outputs.argmax(axis=1).flatten()
            #labels = labels.squeeze().flatten()
            
            

            #outputs = self.model(images)      
            #outputs = outputs.argmax(axis=1).flatten()         
            #labels = labels.squeeze().flatten()
            
            #outputs = outputs.argmax(axis=1).flatten()     
            
            #outputs = outputs.argmax(axis=1).flatten()

            outputs, outputs2 =  outputs.argmax(axis=1).flatten(), outputs.max(axis=1)[0].flatten()

            print(outputs)
            print(outputs.shape)
            sadfasdfas

            
            
            labels = labels.squeeze().flatten()




            
            # stolen from pytorch confusion matrix
            num_classes = len(config_lc.lc_raw_classes.keys())
            unique_mapping = labels.to(torch.long) * num_classes + outputs.to(torch.long)
            bins = torch.bincount(unique_mapping, minlength=num_classes**2) 
            cfm = bins.reshape(num_classes, num_classes)

            return cfm.cpu().numpy()