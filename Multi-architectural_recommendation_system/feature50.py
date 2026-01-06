import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


OUTPUT_VSC = "features_resnet50_with_paths.csv"
DATASET_PATH = "../dataset/data_clean"




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
                image_paths.append(os.path.join(folder, filename))
                labels.append(class_to_idx[class_name])
    return image_paths, labels


def extract_features(model, image_paths, labels, device):
    model.eval()
    model.to(device)
    features = []
    valid_labels = []

    for path, label in tqdm(zip(image_paths, labels), desc="Extracting features", unit="image", total=len(image_paths)):
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(img_tensor).cpu().numpy().squeeze()
            features.append(feat)
            valid_labels.append(label)
        except Exception as e:
            print(f"[Warning] Failed on {path}: {e}")
            continue

    return np.array(features), valid_labels

def main_extract(root_dir = DATASET_PATH, output_csv = OUTPUT_VSC):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    image_paths, labels = load_image_paths_and_labels(root_dir)

    print("Loading ResNet50...")
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()

    print("Extracting ResNet50 features...")
    resnet_features, valid_labels = extract_features(resnet, image_paths, labels, device)

    print("Saving features...")
    df = pd.DataFrame(resnet_features)
    df['label'] = valid_labels
    df['image_path'] = image_paths
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv} | Shape: {df.shape}")

if __name__ == '__main__':
    main_extract()
