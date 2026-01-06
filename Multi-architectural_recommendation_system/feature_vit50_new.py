import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from torchvision.models import resnet50, vit_b_16, ResNet50_Weights, ViT_B_16_Weights

# image preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image_paths_and_labels(root_dir):
    image_paths, labels = [], []
    class_names = sorted(os.listdir(root_dir))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        folder = os.path.join(root_dir, class_name)
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                relative_path = os.path.join(class_name, filename)
                image_paths.append(os.path.join(root_dir, relative_path))
                labels.append(class_to_idx[class_name])
    return image_paths, labels

def extract_features(model, image_paths, labels, device):
    model.eval()
    model.to(device)
    features = []
    valid_paths = []
    valid_labels = []

    for path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc="Extracting features", unit="img"):
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(img_tensor).cpu().numpy().squeeze()
            features.append(feat)
            valid_paths.append(path)
            valid_labels.append(label)
        except Exception as e:
            print(f"[Warning] Failed on {path}: {e}")
            continue

    return np.array(features), valid_paths, valid_labels

def main_extract(root_dir='../dataset/data_clean', output_csv='features_50vit.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_paths, labels = load_image_paths_and_labels(root_dir)

    print("Extracting ResNet50 features...")
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet_features, paths_rn, labels_rn = extract_features(resnet, image_paths, labels, device)

    print("Extracting ViT features...")
    vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    vit.heads = torch.nn.Identity()
    vit_features, paths_vit, labels_vit = extract_features(vit, image_paths, labels, device)

    if paths_rn != paths_vit:
        raise ValueError("ResNet ViT not match please check！")

    print("Concatenating and saving features...")
    features = np.concatenate([resnet_features, vit_features], axis=1)
    df = pd.DataFrame(features)
    df['label'] = labels_rn
    df['image_path'] = [os.path.relpath(p, root_dir) for p in paths_rn]
    df.to_csv(output_csv, index=False)
    print(f"Saved to `{output_csv}`, shape = {df.shape}")

if __name__ == '__main__':
    main_extract()
