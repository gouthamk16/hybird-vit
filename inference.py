import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from vit import HybridVisionTransformer
import matplotlib.pyplot as plt
from pathlib import Path

def load_model(checkpoint_path, device, num_classes):
    """Load a trained model from checkpoint"""
    model = HybridVisionTransformer(
        num_layers=12,
        emb_size=1024,
        num_head=8,
        num_class=num_classes,
        img_size=224
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def preprocess_image(image_path):
    """Preprocess a single image for inference"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, image

def predict(model, image_tensor, class_folders, device, top_k=5):
    """Make a prediction on a single image"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits, embeddings = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        
    # Get top-k predictions
    topk_probs, topk_indices = torch.topk(probabilities, k=top_k)
    topk_probs = topk_probs.cpu().numpy()[0]
    topk_indices = topk_indices.cpu().numpy()[0]
    
    # Map indices to class names
    predictions = []
    for i, idx in enumerate(topk_indices):
        class_name = class_folders[idx]
        predictions.append((class_name, topk_probs[i]))
    
    return predictions, embeddings.cpu().numpy()

def visualize_prediction(image, predictions):
    """Visualize the image and its predictions"""
    plt.figure(figsize=(10, 5))
    
    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')
    
    # Display the predictions
    plt.subplot(1, 2, 2)
    labels = [f"{p[0]}" for p in predictions]
    probs = [p[1] for p in predictions]
    
    bars = plt.barh(range(len(probs)), probs, color='skyblue')
    plt.yticks(range(len(probs)), labels)
    plt.xlabel('Probability')
    plt.title('Top Predictions')
    plt.xlim(0, 1)
    
    # Add text annotations
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f"{probs[i]:.4f}", va='center')
    
    plt.tight_layout()
    plt.show()

def find_similar_cats(embeddings, gallery_embeddings, gallery_paths, top_k=5):
    """Find similar cat images based on embedding similarity"""
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    gallery_embeddings = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
    
    # Calculate cosine similarity
    similarities = np.dot(embeddings, gallery_embeddings.T)[0]
    
    # Get top-k similar images
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_similarities = similarities[top_indices]
    top_paths = [gallery_paths[i] for i in top_indices]
    
    return list(zip(top_paths, top_similarities))

def build_gallery(model, data_dir, device, limit_per_class=5):
    """Build a gallery of cat embeddings for similarity search"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    gallery_embeddings = []
    gallery_paths = []
    gallery_labels = []
    
    data_dir = Path(data_dir)
    class_folders = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    print("Building embedding gallery...")
    for class_idx, class_folder in enumerate(class_folders):
        folder_path = data_dir / class_folder
        image_files = list(folder_path.glob('*.png')) + list(folder_path.glob('*.jpg'))
        
        # Limit number of images per class to save memory
        image_files = image_files[:limit_per_class]
        
        for img_path in image_files:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                _, embedding = model(img_tensor)
                
            gallery_embeddings.append(embedding.cpu().numpy()[0])
            gallery_paths.append(str(img_path))
            gallery_labels.append(class_folder)
            
    return np.array(gallery_embeddings), gallery_paths, gallery_labels

def main():
    parser = argparse.ArgumentParser(description='Inference with cat classification model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='train', help='Directory with cat images')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--top_k', type=int, default=5, help='Show top-k predictions')
    parser.add_argument('--find_similar', action='store_true', help='Find similar cats')
    parser.add_argument('--gallery_limit', type=int, default=5, help='Limit images per class in gallery')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get class folders
    class_folders = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    num_classes = len(class_folders)
    print(f"Found {num_classes} cat classes")
    
    # Load model
    model = load_model(args.checkpoint, device, num_classes)
    print(f"Model loaded from {args.checkpoint}")
    
    # Preprocess image
    image_tensor, image = preprocess_image(args.image_path)
    
    # Make prediction
    predictions, embedding = predict(model, image_tensor, class_folders, device, args.top_k)
    print("\nTop predictions:")
    for i, (class_name, prob) in enumerate(predictions):
        print(f"{i+1}. Class: {class_name}, Probability: {prob:.4f}")
    
    # Visualize prediction
    visualize_prediction(image, predictions)
    
    # Find similar cats if requested
    if args.find_similar:
        print("\nFinding similar cats...")
        gallery_embeddings, gallery_paths, gallery_labels = build_gallery(
            model, args.data_dir, device, args.gallery_limit
        )
        
        similar_cats = find_similar_cats(embedding, gallery_embeddings, gallery_paths, args.top_k)
        
        print("\nMost similar cats:")
        plt.figure(figsize=(15, 3))
        for i, (img_path, similarity) in enumerate(similar_cats):
            plt.subplot(1, args.top_k, i+1)
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(f"Sim: {similarity:.2f}")
            plt.axis('off')
            print(f"{i+1}. {img_path}, Similarity: {similarity:.4f}")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()