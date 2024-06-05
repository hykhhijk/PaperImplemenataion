import torch
import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
# from torchvision import datasets,transforms

import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim // heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.MHA = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)       #dim means input sequence's dim

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)   #example: https://pytorch.org/docs/stable/generated/torch.chunk.html
        # q, k, v = [token for token in qkv]
        q,k,v = qkv
        result = self.MHA(q,k,v, need_weights=True)[0]
        return result

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout = 0.):
        super().__init__()
        layers = []
        layers.append(nn.LayerNorm(dim))
        layers.append(nn.Linear(dim, mlp_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout)),
        layers.append(nn.Linear(mlp_dim, dim)),
        layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout =0.):       #in paper head_dim = dim * 4
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([(Attention(dim, heads, dropout)),(FeedForward(dim, mlp_dim, dropout))]))         
        self.layers = nn.Sequential(*self.layers)

    def forward(self,x):
        for attn, ffn in self.layers:           #이전의 layers를 list에서 iterating하는 방식은 cuda, cpu device 오류남
            x = attn(x) + x
            x = ffn(x) + x
        return x
        
class VisionTransformer(nn.Module):

    def __init__(self, batch_size, dim, depth, heads, mlp_dim, output_dim, img_dim = [3,224,224], patch_dim = [3,56,56], dim_head = 64, dropout =0.):
        super().__init__()
        image_h = img_dim[1]
        image_w = img_dim[2]
        patch_h = patch_dim[1]
        patch_w = patch_dim[2]

        n_patches = (image_h // patch_h) * (image_w // patch_w)
        embedding_dim = img_dim[0] * patch_h * patch_w

        self.patch_dim = patch_dim
        self.img_dim = img_dim
        self.batch_size = batch_size
        self.n_patches = n_patches
        self.embedding_dim = embedding_dim

        #so we flatten the patches and map to D dimensions with a trainable linear projection (Eq. 1).
        self.projection = nn.Sequential(     
            nn.LayerNorm(embedding_dim),                            #layernorm에 대한 언급은 못찾겠음
            nn.Linear(embedding_dim, dim),
            nn.LayerNorm(dim)
        )
        self.cls_token =nn.Parameter(torch.randn(1, dim))
        self.pos_embedding =nn.Parameter(torch.randn(1, n_patches+1, dim))

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.norm = nn.LayerNorm(dim)
        self.classification_head = nn.Linear(dim, output_dim)


    def forward(self, img):
        channels=img.shape[1]

        x = img.unfold(2, self.patch_dim[1], self.patch_dim[2]).unfold(3, self.patch_dim[1], self.patch_dim[2])
        x = x.contiguous().view(self.batch_size, channels, self.n_patches, self.patch_dim[1], self.patch_dim[2])
        patches = x.permute(0, 2, 3, 4, 1)
        x = patches.contiguous().view(self.batch_size, self.n_patches, self.embedding_dim)
        x = self.projection(x)

        cls_tokens = self.cls_token.repeat(self.batch_size, 1, 1)       #(1, dim) -> (b, 1, dim)

        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, :(self.n_patches+1)]

        x = self.transformer(x)
        x = x[:,0]

        x = self.norm(x)                    #is this order right?
        x = self.classification_head(x)
        #많은 구현체에서 norm순서나 유무 vatiation이 많았다

        return x
    

