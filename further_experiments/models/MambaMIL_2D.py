import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.mamba_simple import MambaConfig as SimpleMambaConfig
from models.mamba_simple import Mamba as SimpleMamba

def split_tensor(data, batch_size):
    num_chk = int(np.ceil(data.shape[0] / batch_size))
    return torch.chunk(data, num_chk, dim=0)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class MambaMIL_2D(nn.Module):
    def __init__(self, args):
        super(MambaMIL_2D, self).__init__()
        self.args = args
        self._fc1 = [nn.Linear(args.in_dim, self.args.mambamil_dim)]
        self._fc1 += [nn.GELU()]
        if args.drop_out > 0:
            self._fc1 += [nn.Dropout(args.drop_out)]

        self._fc1 = nn.Sequential(*self._fc1)
        
        self.norm = nn.LayerNorm(self.args.mambamil_dim)
        
        self.layers = nn.ModuleList()
        self.patch_encoder_batch_size = args.patch_encoder_batch_size
        
        
        
        # print('')
        # print(args.mamba_2d_pad_token) 
        # print(args.mamba_2d_patch_size)
        # print('')
        
        # asdfzsdf
        
        # # 512   
        
        
        
        
        
        
        config = SimpleMambaConfig(
            d_model = args.mambamil_dim,
            n_layers = args.mambamil_layer,
            d_state = args.mambamil_state_dim,
            inner_layernorms = args.mambamil_inner_layernorms,
            pscan = args.pscan,
            use_cuda = args.cuda_pscan,
            mamba_2d = True if args.model_type == 'MambaMIL_2D' else False,
            mamba_2d_max_w = 128, # 128 # # 128 # args.mamba_2d_max_w # # args.mamba_2d_max_w # # args.mamba_2d_max_w         
            mamba_2d_max_h = 128, # 128 # # 128 # args.mamba_2d_max_h # # args.mamba_2d_max_h # # args.mamba_2d_max_h  
            mamba_2d_pad_token = args.mamba_2d_pad_token,
            mamba_2d_patch_size = args.mamba_2d_patch_size
        )
        self.layers = SimpleMamba(config)
        self.config = config

        self.n_classes = 11 # args.n_classes # # args.n_classes # # args.n_classes  
        

        self.attention = nn.Sequential(
                nn.Linear(self.args.mambamil_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
        
        #self.classifier = nn.Linear(self.args.mambamil_dim, self.n_classes)       
        
        # self.classifier = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2),  # 1x1 -> 4x4
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),  # 4x4 -> 10x10 (approx)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 11, kernel_size=8, stride=12),  # Adjust kernel/stride to get 128x128
        #     nn.ReLU(),
        # ) 
        
        self.classifier = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1),       # → (256, 4, 4)
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4),       # → (128, 16, 16)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # → (64, 32, 32)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # → (32, 64, 64)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 11, kernel_size=4, stride=2, padding=1),   # → (11, 128, 128)
        ) 
        
        
        
        
        # sadfasdf
        # #batch_size = 4  # example
        # embedding = torch.randn(batch_size, 512)  # shape: [batch_size, 512]

        # # Step 1: Project to some lower spatial representation
        # fc = nn.Linear(512, 11 * 8 * 8)
        # x = fc(embedding)  # shape: [batch_size, 11*8*8]
        # x = x.view(batch_size, 11, 8, 8)  # [batch_size, 11, 8, 8]

        # # Step 2: Upsample to 128x128
        # upsample = nn.Sequential(
        #     nn.ConvTranspose2d(11, 11, kernel_size=4, stride=2, padding=1),  # 8x8 → 16x16
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(11, 11, kernel_size=4, stride=2, padding=1),  # 16x16 → 32x32
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(11, 11, kernel_size=4, stride=2, padding=1),  # 32x32 → 64x64
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(11, 11, kernel_size=4, stride=2, padding=1),  # 64x64 → 128x128
        # )

        # output = upsample(x)  # shape: [batch_size, 11, 128, 128]
        # sadfasdf
        
        
        
        
        
        
        # reshape to (batch, channels, height, width)   
        #x = x.view(1, 512, 1, 1) 

        # deconv = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2),  # 1x1 -> 4x4
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),  # 4x4 -> 10x10 (approx)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 11, kernel_size=8, stride=12),  # Adjust kernel/stride to get 128x128
        #     nn.ReLU(),
        # )

        #out = deconv(x) 
        
        
        
        
        
        self.survival = args.survival

        if args.pos_emb_type == 'linear':
            self.pos_embs = nn.Linear(2, args.mambamil_dim)
            self.norm_pe = nn.LayerNorm(args.mambamil_dim)
            self.pos_emb_dropout = nn.Dropout(args.pos_emb_dropout)
        else:
            self.pos_embs = None

        self.apply(initialize_weights)

    def forward(self, x, coords=None):  
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)   # (1, num_patch, feature_dim)
        h = x.float()  # [1, num_patch, feature_dim]

        h = self._fc1(h)  # [1, num_patch, mamba_dim];   project from feature_dim -> mamba_dim

        # Add Pos_emb
        if self.args.pos_emb_type == 'linear':
            pos_embs = self.pos_embs(coords)
            h = h + pos_embs.unsqueeze(0)
            h = self.pos_emb_dropout(h)

        h = self.layers(h, coords, self.pos_embs)

        h = self.norm(h)   # LayerNorm
        A = self.attention(h) # [1, W, H, 1]

        if len(A.shape) == 3:
            A = torch.transpose(A, 1, 2)
        else:  
            A = A.permute(0,3,1,2)
            A = A.view(1,1,-1)
            h = h.view(1,-1,self.config.d_model)

        A = F.softmax(A, dim=-1)  # [1, 1, num_patch]  # A: attention weights of patches      
        h = torch.bmm(A, h) # [1, 1, 512] , weighted combination to obtain slide feature
        h = h.squeeze(0)  # [1, 512], 512 is the slide dim



        # print(h.shape)           

        # print(x.shape[0]) 

        # sadfsadf

        # # torch.Size([1, 512])   
        # # 32 



        # # torch.Size([1, 512])
        



        

        #print(h.shape) 

        #sadfasdf
        
        
        # reshape to (batch, channels, height, width) 
        #x = x.view(1, 512, 1, 1)
        #h = h.view(1, 512, 1, 1)



        h = h.expand(x.shape[0], -1)
        
        
        
        # reshape to (batch, channels, height, width) 
        #x = x.view(1, 512, 1, 1)
        #h = h.view(1, 512, 1, 1)

        #h = h.view(1, 512, 1, 1)
        h = h.view(x.shape[0], 512, 1, 1)
        
        

        # deconv = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2),  # 1x1 -> 4x4
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),  # 4x4 -> 10x10 (approx)
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 11, kernel_size=8, stride=12),  # Adjust kernel/stride to get 128x128
        #     nn.ReLU(),
        # )

        # out = deconv(x)        


        logits = self.classifier(h)  # # [1, n_classes]       
        #Y_prob = F.softmax(logits, dim=1) 
        #Y_hat = torch.topk(logits, 1, dim=1)[1]
        #results_dict = None

        if self.survival:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None # # same return as other models   

        #return logits, Y_prob, Y_hat, results_dict, None # same return as other models
        return logits
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers  = self.layers.to(device)
        
        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)
