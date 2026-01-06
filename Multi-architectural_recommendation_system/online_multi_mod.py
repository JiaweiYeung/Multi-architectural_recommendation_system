import os
import numpy as np
import requests
import torch
from PIL import Image
from io import BytesIO
from hashlib import md5
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlencode
from part1_model.trainer_deeper import DeepResMLPClassifier

# ==== Google API ==== Maximum is 10/time and 100times/day
GOOGLE_API_KEY = "AIzaSyDT7hLJtTVh6lxuaSY0pH_HZq9G7DyspK8"
GOOGLE_CSE_ID = "71b87123abc9f4707"

# ==== Text search ====
def search_similar_images(style, location=None, period=None, num=10):

    query_parts = [style]
    if location:
        query_parts.append(location)
    if period:
        query_parts.append(period)
    query = " ".join(query_parts)

    params = {
        "q": query,
        "cx": GOOGLE_CSE_ID,
        "key": GOOGLE_API_KEY,
        "searchType": "image",
        "num": num
    }
    url = f"https://www.googleapis.com/customsearch/v1?{urlencode(params)}"
    response = requests.get(url).json()

    results = []
    if "items" in response:
        for item in response["items"]:
            results.append({
                "link": item["link"],
                "source": item.get("image", {}).get("contextLink", ""),
                "title": item.get("title", "")
            })

    print(f"[DEBUG] Retrieved {len(results)} Google image links")

    return results

# ==== Image Preprocessing ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== Load Class Names ====
def load_class_names(data_dir):
    if data_dir and os.path.exists(data_dir):
        return sorted(os.listdir(data_dir))
    else:
        return [f"Style_{i}" for i in range(25)]

# ==== Predicting Style ====
def predict_image_style(image_path, model_path, data_dir=None, feature_dim=2818):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = load_class_names(data_dir)
    num_classes = len(class_names)

    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get("model_config", None)
    if model_config is not None:
        classifier = DeepResMLPClassifier(**model_config).to(device)
        classifier.load_state_dict(checkpoint["model_state_dict"])
    else:
        classifier = DeepResMLPClassifier(input_dim=feature_dim, num_classes=num_classes)
        classifier.load_state_dict(checkpoint)
    classifier.eval()

    backbone_rn = resnet50(weights=ResNet50_Weights.DEFAULT)
    backbone_rn.fc = nn.Identity()
    backbone_rn.to(device).eval()

    backbone_vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    backbone_vit.heads = nn.Identity()
    backbone_vit.to(device).eval()

    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat_rn = backbone_rn(img_tensor).cpu().numpy()
            feat_vit = backbone_vit(img_tensor).cpu().numpy()
            features = np.concatenate([feat_rn, feat_vit], axis=1)

            input_tensor = torch.tensor(features, dtype=torch.float32).to(device)
            outputs = classifier(input_tensor)

            predicted_idx = outputs.argmax(1).item()
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence = probabilities[predicted_idx].item() * 100
            predicted_style = class_names[predicted_idx]

            print("\n== Prediction Results ==")
            print(f"Architectural style: {predicted_style}")
            print(f"Confidence: {confidence:.2f}%")

            return predicted_style, features
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, 0

# ==== Online Image Similarity Comparison ====
def compare_with_similar_images(original_feat, image_urls, top_k=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone_rn = resnet50(weights=ResNet50_Weights.DEFAULT)
    backbone_rn.fc = nn.Identity()
    backbone_rn.to(device).eval()

    backbone_vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    backbone_vit.heads = nn.Identity()
    backbone_vit.to(device).eval()

    seen_hashes = set()
    results = []

    for item in image_urls:
        try:
            response = requests.get(item["link"], timeout=5)
            img = Image.open(BytesIO(response.content)).convert('RGB')

            img_bytes = BytesIO()
            img.save(img_bytes, format='JPEG')
            hash_val = md5(img_bytes.getvalue()).hexdigest()
            if hash_val in seen_hashes:
                continue
            seen_hashes.add(hash_val)

            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                feat_rn = backbone_rn(img_tensor).cpu().numpy()
                feat_vit = backbone_vit(img_tensor).cpu().numpy()
                feat = np.concatenate([feat_rn, feat_vit], axis=1)
                sim = cosine_similarity(original_feat, feat)[0][0]
                results.append((item["link"], item["source"], sim))

        except Exception:
            continue

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]

# ==== Test Entry ====
if __name__ == "__main__":
    test_image = "../images/test_image/test_img_1.png"
    model_path = "../part1_model/best_grid_50vit_deep.pth"
    data_dir = "../dataset/data_clean"

    style, feat = predict_image_style(test_image, model_path, data_dir)
    if style and feat is not None:
        candidates = search_similar_images(style, num=10)
        results = compare_with_similar_images(feat, candidates, top_k=10)

        print("\n== Top Similar Online Images ==")
        for idx, (url, source, sim) in enumerate(results, 1):
            print(f"{idx}. Similarity: {sim:.4f}")
            print(f"   URL: {url}")
            print(f"   Source: {source}")
