import math
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Encoder class for the ViT model

    Args:
    num_hidden_layers (int): Number of hidden layers
    hidden_size (int): Hidden size
    intermediate_size (int): Intermediate size
    num_attention_heads (int): Number of attention heads. Defaults to 4.
    attention_probs_dropout_prob (float): Dropout probability for attention. Defaults to 0.0.
    hidden_dropout_prob (float): Dropout probability for hidden layers. Defaults to 0.0.
    qkv_bias (bool, optional): Whether to use bias for query, key and value. Defaults to True.

    Returns:
    X (Tensor): Tensor containing the output of the encoder
    """
    def __init__(self, num_hidden_layers:int, hidden_size:int, intermediate_size:int, num_attention_heads:int=4, attention_probs_dropout_prob:float=0.0, hidden_dropout_prob:float=0.0, qkv_bias:bool=True):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_hidden_layers):
            block = EncoderBlock(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob, qkv_bias).to(device)
            self.blocks.append(block)


    def forward(self, X):
        for block in self.blocks:
            X = block(X)
        return X


class EncoderBlock(nn.Module):
    """
    Encoder block for the ViT model

    Args:
    hidden_size (int): Hidden size.
    intermediate_size (int): Intermediate size.
    num_attention_heads (int): Number of attention heads. Defaults to 4.
    attention_probs_dropout_prob (float): Dropout probability for attention. Defaults to 0.0.
    hidden_dropout_prob (float): Dropout probability for hidden layers. Defaults to 0.0.
    qkv_bias (bool, optional): Whether to use bias for query, key and value. Defaults to True.

    Returns:
    X (Tensor): Tensor containing the output of the encoder block
    """
    def __init__(self, hidden_size:int, intermediate_size:int, num_attention_heads:int=4, attention_probs_dropout_prob:float=0.0, hidden_dropout_prob:float=0.0, qkv_bias=True):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.MHA = MultiHeadAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob, qkv_bias).to(device)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, intermediate_size, hidden_dropout_prob).to(device)
    
    def forward(self, X):
        MHA_X = self.norm1(X)
        attention_output = self.MHA(MHA_X)

        X = X + attention_output

        NORM_X = self.norm2(X)
        mlp_output = self.mlp(NORM_X)

        X = X + mlp_output

        return X


class MLP(nn.Module):
    """
    Multi Layer Perceptron for the ViT model

    Args:
    hidden_size (int): Hidden size
    intermediate_size (int): Intermediate size
    hidden_dropout_prob (float): Dropout probability for hidden layers. Defaults to 0.0.

    Returns:
    X (Tensor): Tensor containing the output of the MLP
    """
    def __init__(self, hidden_size:int, intermediate_size:int, hidden_dropout_prob:int=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.GELU = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, X):
        X = self.fc1(X)
        X = self.GELU(X)
        X = self.fc2(X)
        X = self.dropout(X)
        return X


class Scaled_Dot_Product_Attention(nn.Module):
    """
    Scale Dot Product Attention for the ViT model

    Args:
    hidden_size (int): Hidden size
    attention_head_size (int): Attention head size
    dropout_prob (float): Dropout probability for attention. Defaults to 0.0.
    bias (bool, optional): Whether to use bias for query, key and value. Defaults to True.

    Returns:
    attention_output (Tensor): Tensor containing the attention output
    attention_probs (Tensor): Tensor containing the attention probabilities
    """
    def __init__(self, hidden_size:int, attention_head_size:int, dropout_prob:float=0.0, bias=True):
        super(Scaled_Dot_Product_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size

        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, X):
        query = self.query(X)
        key = self.key(X)
        value = self.value(X)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention for the ViT model

    Args:
    hidden_size (int): Hidden size
    num_attention_heads (int): Number of attention heads. Defaults to 4.
    attention_probs_dropout_prob (float): Dropout probability for attention. Defaults to 0.0.
    hidden_dropout_prob (float): Dropout probability for hidden layers. Defaults to 0.0.
    qkv_bias (bool, optional): Whether to use bias for query, key and value. Defaults to True.

    Returns:
    attention_output (Tensor): Tensor containing the attention output
    """
    def __init__(self, hidden_size:int, num_attention_heads:int=4, attention_probs_dropout_prob:float=0.0, hidden_dropout_prob:float=0.0, qkv_bias:bool=True):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.qkv_bias = qkv_bias
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = Scaled_Dot_Product_Attention(self.hidden_size, self.attention_head_size, attention_probs_dropout_prob, bias=self.qkv_bias).to(device)
            self.heads.append(head)
        
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, X):
        attention_outputs = [head(X) for head in self.heads]
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        return attention_output


class PatchEmbeddings(nn.Module):
    """
    Patch Embeddings for the ViT model

    Args:
    image_size (int): Image size
    hidden_size (int): Hidden size
    patch_size (int): Patch size. Defaults to 16.
    num_channels (int): Number of channels. Defaults to 3(RGB).

    Returns:
    X (Tensor): Tensor containing the output of the patch embeddings
    """
    def __init__(self, image_size:int, hidden_size:int,  patch_size:int=16, num_channels:int=3):
        super(PatchEmbeddings, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.num_patches = (self.image_size // self.patch_size) * (self.image_size // self.patch_size)
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
    
    def forward(self, X):
        X = self.projection(X)
        X = X.flatten(2)
        X = X.transpose(-1, -2)
        return X


class Embeddings(nn.Module):
    """
    Embeddings for the ViT model

    Args:
    hidden_size (int): Hidden size
    image_size (int): Image size
    patch_size (int): Patch size. Defaults to 16.
    num_channels (int): Number of channels. Defaults to 3(RGB).
    hidden_dropout_prob (float): Dropout probability for hidden layers. Defaults to 0.0.

    Returns:
    X (Tensor): Tensor containing the output of the embeddings
    """
    def __init__(self, hidden_size:int, image_size:int, patch_size:int=16, num_channels:int=3, hidden_dropout_prob:float=0.0):
        super(Embeddings, self).__init__()
        self.patch_embeddings = PatchEmbeddings(image_size, hidden_size, patch_size, num_channels).to(device)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches+1, hidden_size))
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, X):
        X = self.patch_embeddings(X)
        batch_size, _, _ = X.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        X = torch.cat((cls_tokens, X), dim=1)
        X = X + self.position_embeddings
        X = self.dropout(X)
        return X


class ViT(nn.Module):
    """
    Vision Transformer model

    Args:
    image_size (int): Image size
    hidden_size (int): Hidden size
    num_hidden_layers (int): Number of hidden layers
    intermediate_size (int): Intermediate size
    num_classes (int): Number of classes
    num_attention_heads (int): Number of attention heads. Defaults to 4.
    hidden_dropout_prob (float): Dropout probability for hidden layers. Defaults to 0.0.
    attention_probs_dropout_prob (float): Dropout probability for attention. Defaults to 0.0.
    num_channels (int): Number of channels. Defaults to 3(RGB).
    patch_size (int): Patch size. Defaults to 16.
    qkv_bias (bool, optional): Whether to use bias for query, key and value. Defaults to True.

    Returns:
    logits (Tensor): Tensor containing the logits
    """
    def __init__(self, image_size:int, hidden_size:int, num_hidden_layers:int, intermediate_size, num_classes:int, num_attention_heads:int=4, 
                 hidden_dropout_prob:float=0.0, attention_probs_dropout_prob:float=0.0, num_channels:int=3, patch_size:int=16, qkv_bias:bool=True):
        super(ViT, self).__init__()
        self.embeddings = Embeddings(hidden_size, image_size, patch_size, num_channels, hidden_dropout_prob).to(device)
        self.encoder = Encoder(num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, 
                               attention_probs_dropout_prob, hidden_dropout_prob, qkv_bias).to(device)
        self.classifier = nn.Linear(hidden_size, num_classes).to(device)
        self.apply(self._init_weights)

    def forward(self, X):
        embedding_output = self.embeddings(X)
        encoder_output = self.encoder(embedding_output)
        logits = self.classifier(encoder_output[:, 0, :])
        return logits
    
    def _init_weights(self, module, initializer_range:float=0.02):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(module.position_embeddings.data.to(torch.float32), 
                                                                    mean=0.0, 
                                                                    std=initializer_range).to(module.position_embeddings.dtype)
            module.cls_token.data = nn.init.trunc_normal_(module.cls_token.data.to(torch.float32), 
                                                          mean=0.0, 
                                                          std=initializer_range).to(module.cls_token.dtype)


import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
image_size = 224
batch_size = 32
num_epochs = 10
learning_rate = 3e-4
hidden_size = 48
num_hidden_layers = 4
num_attention_heads = 4
intermediate_size = hidden_size * 4
patch_size = 16
dropout = 0.1
train_dir = "/content/train2/training data 2"

# Data transforms
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# Dataset and Dataloader
dataset = ImageFolder(root=train_dir, transform=transform)
num_classes = len(dataset.classes)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Model
model = ViT(image_size=image_size, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size, num_classes=num_classes, 
            num_attention_heads=num_attention_heads, hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout, patch_size=patch_size).to(device)

# Loss and optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += torch.sum(predictions == labels).item()
        print(total, correct)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

# Save model
torch.save(model.state_dict(), "vit_cat_classifier.pth")
print("Training complete. Model saved.")