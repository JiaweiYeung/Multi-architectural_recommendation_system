import os
import numpy as np
import pandas as pd
import torch
from PIL import Image

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from part1_model.trainer_deeper import DeepResMLPClassifier

ROOT_DIR = "../dataset/data_clean"

# ===== Load Category Index Mapping =====
def load_label_mapping(root_dir=ROOT_DIR):
    class_names = sorted(os.listdir(root_dir))
    idx_to_class = {idx: name for idx, name in enumerate(class_names)}
    return idx_to_class

# ===== predictor function =====
def predict_architecture_style(image_path,model_path="../part1_model/best_grid_50vit_deep.pth",
    feature_csv="../part1_model/features_50vit.csv",root_dir=ROOT_DIR,return_feature=False):
    # Step 1: Get feature dimensions and number of classes (to ensure idx_to_class is in the right order)
    df = pd.read_csv(feature_csv)
    idx_to_class = load_label_mapping(root_dir)

    # Step 2: Load trained model + config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint["model_config"]
    model = DeepResMLPClassifier(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Step 3: image process
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Step 4: Extracting ResNet + ViT features
    with torch.no_grad():
        # ResNet50
        backbone_rn = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone_rn.fc = torch.nn.Identity()
        backbone_rn.to(device)
        backbone_rn.eval()
        feat_rn = backbone_rn(img_tensor).cpu().numpy()

        # ViT
        backbone_vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        backbone_vit.heads = torch.nn.Identity()
        backbone_vit.to(device)
        backbone_vit.eval()
        feat_vit = backbone_vit(img_tensor).cpu().numpy()

        # Patchwork Final Features
        feature = np.concatenate([feat_rn, feat_vit], axis=1)  # [1, 2818]
        input_tensor = torch.tensor(feature, dtype=torch.float32).to(device)

        # Step 5: predict
        output = model(input_tensor).softmax(dim=1)
        predicted_label = output.argmax().item()
        confidence = output.max().item()

    # Step 6: mapping class
    style_name = idx_to_class[predicted_label]
    print(f"Predicted Style: {style_name} (ID: {predicted_label}, Confidence: {confidence:.4f})")
    # return style_name, predicted_label, confidence
    if return_feature:
        return style_name, predicted_label, confidence, feature
    else:
        return style_name, predicted_label, confidence


# ===== test run current py file =====
if __name__ == '__main__':
    predict_architecture_style("../images/test_image/test_img_1.png")
