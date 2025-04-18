# MIT License

# Copyright (c) 2024 Shubh Mishra

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import models


def PositionEmbedding(seq_len, emb_size):
    embeddings = torch.ones(seq_len, emb_size)
    for i in range(seq_len):
        for j in range(emb_size):
            embeddings[i][j] = np.sin(i / (pow(10000, j / emb_size))) if j % 2 == 0 else np.cos(i / (pow(10000, (j - 1) / emb_size)))
    return torch.tensor(embeddings)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size=224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_size))
        self.pos_embed = nn.Parameter(PositionEmbedding((img_size // patch_size)**2 + 1, emb_size))
    
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)    

        cls_token = repeat(self.cls_token, ' () s e -> b s e', b=b)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        return x


class MultiHead(nn.Module):
  def __init__(self, emb_size, num_head):
    super().__init__()
    self.emb_size = emb_size
    self.num_head = num_head
    self.key = nn.Linear(emb_size, emb_size)
    self.value = nn.Linear(emb_size, emb_size)
    self.query = nn.Linear(emb_size, emb_size) 
    self.att_dr = nn.Dropout(0.1)
  def forward(self, x):
    k = rearrange(self.key(x), 'b n (h e) -> b h n e', h=self.num_head)
    q = rearrange(self.query(x), 'b n (h e) -> b h n e', h=self.num_head)
    v = rearrange(self.value(x), 'b n (h e) -> b h n e', h=self.num_head)

    wei = q@k.transpose(3,2)/self.num_head ** 0.5    
    wei = F.softmax(wei, dim=2)
    wei = self.att_dr(wei)

    out = wei@v

    out = rearrange(out, 'b h n e -> b n (h e)')
    return out

  
class FeedForward(nn.Module):
  def __init__(self, emb_size):
    super().__init__()
    self.ff = nn.Sequential(
        nn.Linear(emb_size, 4*emb_size),
        nn.Linear(4*emb_size, emb_size)
    )
  def forward(self, x):
    return self.ff(x)
  
class Block(nn.Module):
  def __init__(self,emb_size, num_head):
    super().__init__()
    self.att = MultiHead(emb_size, num_head)
    self.ll =   nn.LayerNorm(emb_size)
    self.dropout = nn.Dropout(0.1)
    self.ff = FeedForward(emb_size)
  def forward(self, x):
    x = x + self.dropout(self.att(self.ll(x)))  # self.att(x): x -> (b , n, emb_size) 
    x = x + self.dropout(self.ff(self.ll(x)))
    return x
  
class VissionTransformer(nn.Module):
  def __init__(self, num_layers, img_size, emb_size, patch_size, num_head, num_class):
    super().__init__()
    self.attention = nn.Sequential(*[Block(emb_size, num_head) for _ in range(num_layers)])
    self.patchemb = PatchEmbedding(patch_size=patch_size, img_size=img_size)
    self.ff = nn.Linear(emb_size, num_class)

  def forward(self, x):     # x -> (b, c, h, w)
    embeddings = self.patchemb(x)    
    x = self.attention(embeddings)   
    x = self.ff(x[:, 0, :])
    return x
  
# New Hybrid model using ResNet backbone
class HybridVisionTransformer(nn.Module):
    def __init__(self, num_layers=12, emb_size=1024, num_head=8, num_class=200000, img_size=224):
        super().__init__()
        
        # ResNet backbone (more efficient for feature extraction)
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avg pool and FC
        
        # Get the output size from ResNet
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat_size = self.backbone(dummy).shape
            seq_len = (feat_size[2] * feat_size[3])
        
        # Project ResNet features to transformer dimension
        self.projection = nn.Sequential(
            nn.Conv2d(feat_size[1], emb_size, kernel_size=1),
            Rearrange('b e h w -> b (h w) e'),
        )
        
        # Add CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, emb_size))
        
        # Transformer layers
        self.transformer = nn.Sequential(*[Block(emb_size, num_head) for _ in range(num_layers)])
        
        # Classification head - using ArcFace-style normalization for better class separation
        self.norm = nn.LayerNorm(emb_size)
        self.embedding_layer = nn.Linear(emb_size, 2048)  # Generate embedding
        self.classifier = nn.Linear(2048, num_class)      # Classification layer
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Extract features using ResNet
        x = self.backbone(x)
        
        # Project features to embedding dimension
        x = self.projection(x)
        
        # Add CLS token
        b = x.shape[0]
        cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Pass through transformer blocks
        x = self.transformer(x)
        
        # Use CLS token for classification
        x = self.norm(x[:, 0])
        
        # Generate embedding
        embedding = self.embedding_layer(x)
        embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalization
        
        # Classification logits
        logits = self.classifier(self.dropout(embedding))
        
        return logits, embedding

# Configuration for large-scale cat classification
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_layers = 12
emb_size = 1024
num_head = 8
num_class = 200000  # Adjusted for your 200k classes
patch_size = 16
img_size = 224

# Initialize the hybrid model
model = HybridVisionTransformer(
    num_layers=num_layers,
    emb_size=emb_size,
    num_head=num_head,
    num_class=num_class,
    img_size=img_size
).to(device)

# The original ViT model is kept for reference
vit_model = VissionTransformer(
    num_layers=8,
    img_size=224,
    emb_size=768,
    patch_size=16,
    num_head=6,
    num_class=10
).to(device)

if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    x = x.type(torch.FloatTensor).to(device)
    
    # Test the hybrid model
    logits, embeddings = model(x)
    print(f"Hybrid model output - Logits shape: {logits.shape}, Embeddings shape: {embeddings.shape}")
    
    # Test the original ViT model
    vit_output = vit_model(x)
    print(f"Original ViT model output shape: {vit_output.shape}")