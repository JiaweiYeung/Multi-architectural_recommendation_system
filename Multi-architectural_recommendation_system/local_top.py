import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import normalize

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from part1_model.trainer_deeper import DeepResMLPClassifier
from part2_localsearch.local_search import load_label_mapping
from part2_localsearch.local_search import predict_architecture_style


def retrieve_similar_images(
    query_image_path,
    model_path="../part1_model/best_grid_50vit_deep.pth",
    feature_csv="../part1_model/features_50vit.csv",
    root_dir="../dataset/data_clean",
    top_k=10,
    shared_feature=None,
    shared_pred_label=None,
    shared_conf=None,
    shared_style = None

):
    df = pd.read_csv(feature_csv)
    feature_cols = [col for col in df.columns if col not in ['label', 'image_path']]
    features = df[feature_cols].values
    features = normalize(features)
    paths = df['image_path'].values
    labels = df['label'].values

    idx_to_class = load_label_mapping(root_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")

    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint["model_config"]
    model = DeepResMLPClassifier(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if shared_feature is None or shared_pred_label is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        backbone_rn = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone_rn.fc = torch.nn.Identity()
        backbone_rn.to(device).eval()
        backbone_vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        backbone_vit.heads = torch.nn.Identity()
        backbone_vit.to(device).eval()
        img = Image.open(query_image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat_rn = backbone_rn(img_tensor).cpu().numpy()
            feat_vit = backbone_vit(img_tensor).cpu().numpy()
            query_feat = np.concatenate([feat_rn, feat_vit], axis=1)
            query_feat = normalize(query_feat)
            input_tensor = torch.tensor(query_feat, dtype=torch.float32).to(device)
            output = model(input_tensor).softmax(dim=1)
            pred_label = output.argmax().item()
            pred_conf = output.max().item()
    else:
        query_feat = normalize(shared_feature)
        pred_label = shared_pred_label
        pred_conf = shared_conf

    query_vec = query_feat.squeeze()
    same_mask = labels == pred_label
    other_mask = labels != pred_label

    features_same = features[same_mask]
    paths_same = paths[same_mask]
    labels_same = labels[same_mask]

    features_other = features[other_mask]
    paths_other = paths[other_mask]
    labels_other = labels[other_mask]

    similarities_same = np.dot(features_same, query_vec)
    similarities_other = np.dot(features_other, query_vec)

    k_same = top_k // 2
    k_other = top_k - k_same

    topk_same = np.argsort(similarities_same)[::-1][:k_same]
    topk_other = np.argsort(similarities_other)[::-1][:k_other]

    results = []

    for i, idx in enumerate(topk_same):
        results.append({
            'rank': i + 1,
            'style': idx_to_class[labels_same[idx]],
            'label': int(labels_same[idx]),
            'path': paths_same[idx],
            'similarity': float(similarities_same[idx]),
            'source': 'same'
        })

    for i, idx in enumerate(topk_other):
        results.append({
            'rank': k_same + i + 1,
            'style': idx_to_class[labels_other[idx]],
            'label': int(labels_other[idx]),
            'path': paths_other[idx],
            'similarity': float(similarities_other[idx]),
            'source': 'other'
        })

    print(f"\nQuery Prediction: {idx_to_class[pred_label]} (Conf: {pred_conf:.4f})")
    print(f"Top-{top_k} similar images (half same class, half other):")
    for r in results:
        print(f"{r['rank']}. [{r['similarity']:.4f}] ({r['source']}) {r['style']} | {r['path']}")

    return results



def local_top_api(image_path):
    style_, label_, conf_, feature_ = predict_architecture_style(
        image_path, return_feature=True
    )
    results_ = retrieve_similar_images(
        image_path,
        shared_feature=feature_,
        shared_pred_label=label_,
        shared_conf=conf_,
        shared_style=style_
    )
    return results_

# ===== test =====
if __name__ == '__main__':

    style, label, conf, feature = predict_architecture_style(
        "../images/test_image/test_img_1.png", return_feature=True
    )
    results = retrieve_similar_images(
        "../images/test_image/test_img_1.png",
        shared_feature=feature,
        shared_pred_label=label,
        shared_conf=conf
    )
    print("local_results",results)
    # for r in results:
    #     print(r)

