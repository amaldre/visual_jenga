import torch
import torchvision.transforms as T
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import timm
import torch.nn.functional as F

# Load CLIP (ViT-B/32)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load DINO (ViT-S/8)
dino_model = timm.create_model('vit_small_patch8_224_dino', pretrained=True)
dino_model.eval()

# Image transform
dino_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Similarity function (cosine similarity)
def cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=-1)

# Encode image using CLIP
def get_clip_embedding(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features / features.norm(dim=-1, keepdim=True)

def get_dino_embedding(image: Image.Image):
    tensor = dino_transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]
    with torch.no_grad():
        features = dino_model.forward_features(tensor)  # Returns tensor
    # Take [CLS] token (first token)
    if features.ndim == 3:
        features = features[:, 0, :]  # shape: [1, dim]
    return features / features.norm(dim=-1, keepdim=True)


def combined_score(c_new_list, c_orig):
    N = len(c_new_list)
    
    clip_sims = []
    dino_sims = []
    
    clip_orig = get_clip_embedding(c_orig)
    dino_orig = get_dino_embedding(c_orig)

    for c_new in c_new_list:
        clip_feat = get_clip_embedding(c_new)
        dino_feat = get_dino_embedding(c_new)

        clip_sims.append(cosine_similarity(clip_feat, clip_orig))
        dino_sims.append(cosine_similarity(dino_feat, dino_orig))
    
    clip_mean = torch.stack(clip_sims).mean()
    dino_mean = torch.stack(dino_sims).mean()

    return 1 - (clip_mean * dino_mean).item()