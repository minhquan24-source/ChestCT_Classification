'''
Reference:
Phan, H., Le, C., Le, V., He, Y., & Nguyen, A. (2024). 
Fast and Interpretable Face Identification for Out-Of-Distribution Data Using Vision Transformers. 
https://doi.org/10.1109/WACV57701.2024.00618  
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from einops import rearrange, repeat
from resnet18 import resnet_face18

# Minimum number of patches required 
MIN_NUM_PATCHES = 16

class ArcFace(nn.Module):
    '''
    Description: ArcFace loss function (angular margin)
    '''
    def __init__(self, in_features, out_features, device_id=None, s=64.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # Compute cosine similarity
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # Compute sine and phi(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding of the labels
        one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1).long(), 1)

        # Apply ArcFace margin to output
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

class Residual(nn.Module):
    '''
    Description: Residual connection for the transformer
    '''
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        attn, out = self.fn(x, **kwargs)
        return attn, out + x

class PreNorm(nn.Module):
    '''
    Description: Pre-normalization for the transformer
    '''
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        attn, out = self.fn(self.norm(x), **kwargs) if isinstance(self.fn, Attention) else (None, self.fn(self.norm(x), **kwargs))
        return attn, out

class FeedForward(nn.Module):
    '''
    Description: Feed-forward network for the transformer
    '''
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    '''
    Description: Multi-head self-attention for the transformer
    '''
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None:
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, -torch.finfo(dots.dtype).max)

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return attn, self.to_out(out)

class Transformer(nn.Module):
    '''
    Description: Transformer model
    '''
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]) for _ in range(depth)
        ])

    def forward(self, x, mask=None):
        attns = []
        for attn, ff in self.layers:
            layer_attn, x = attn(x, mask=mask)
            attns.append(layer_attn)
            _, x = ff(x)
        return x, attns

class HybridViT(nn.Module):
    '''
    Description: Hybrid Vision Transformer model
    '''
    def __init__(self, GPU_ID, num_class, image_size, patch_size, dim, depth, heads, mlp_dim, channels=1, out_dim=512, remove_pos=False):
        super(HybridViT, self).__init__()

        self.face_model = self.load_face_model()
        self.remove_pos = remove_pos
        self.patch_size = patch_size

        num_patches = (image_size // patch_size) ** 2
        assert num_patches > MIN_NUM_PATCHES, f'Number of patches ({num_patches}) is too small for effective attention.'

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) if not remove_pos else None
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = Transformer(dim, depth, heads, dim // heads, mlp_dim, dropout=0.1)
        self.mlp_head = nn.LayerNorm(out_dim)
        self.loss = ArcFace(in_features=out_dim, out_features=num_class, device_id=GPU_ID)

    def load_face_model(self):
        '''
        Description: Load the pre-trained resnet18 
        '''
        print('Loading face model...')
        # Use resnet18 model with grayscale image
        facemodel = resnet_face18(False, grayscale=True)
        # Load the pre-trained weights
        state_dict = torch.load('./ViT/resnet18_110.pth', map_location=torch.device('cpu'))
        # Remove the 'module.' prefix from the keys
        facemodel.load_state_dict({k[7:]: v for k, v in state_dict.items()})
        return facemodel

    def forward(self, img, label=None, mask=None):
        # Extract features from input image
        feature = self.face_model(img)['embedding']
        b, c, h, w = feature.size()
        # Reshape the features to patches
        x = feature.view(b, c, -1).transpose(1, 2)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # Concatenate classification tokens with the patches
        x = torch.cat((cls_tokens, x), dim=1)
        # Add positional embedding
        if self.pos_embedding is not None:
            x += self.pos_embedding[:, :(x.size(1))]

        # Apply dropout
        x = self.dropout(x)
        # Apply transformer
        x, _ = self.transformer(x, mask)
        # Apply MLP head
        emb = self.mlp_head(x.mean(dim=1))

        # Apply loss function if label is provided
        return self.loss(emb, label) if label is not None else emb