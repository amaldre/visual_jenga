import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import numpy as np


def cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=-1)

# Get clip and Dino Embedding with 224*224 input images
def get_clip_embedding(image, clip_model, clip_processor):
    inputs = clip_processor(images=image, return_tensors="pt")

    device = next(clip_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)

    return features / features.norm(dim=-1, keepdim=True)


def get_dino_embedding(image, dino_model):
    dino_model.eval()
    dino_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    tensor = dino_transform(image).unsqueeze(0)
    device = next(dino_model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        features = dino_model.forward_features(tensor)

    if features.ndim == 3:
        features = features[:, 0, :]

    return features / features.norm(dim=-1, keepdim=True)


# Combined similarity score using Dino and Clip embeddings
def combined_score(c_new_list, c_orig, clip_model, clip_processor, dino_model):
    N = len(c_new_list)
    
    clip_sims = []
    dino_sims = []
    
    clip_orig = get_clip_embedding(c_orig, clip_model, clip_processor)
    dino_orig = get_dino_embedding(c_orig, dino_model) 

    for c_new in c_new_list:
        clip_feat = get_clip_embedding(c_new, clip_model, clip_processor)
        dino_feat = get_dino_embedding(c_new, dino_model)

        clip_sims.append(cosine_similarity(clip_feat, clip_orig))
        dino_sims.append(cosine_similarity(dino_feat, dino_orig))
    
    clip_mean = torch.stack(clip_sims).mean()
    dino_mean = torch.stack(dino_sims).mean()

    return 1 - (clip_mean * dino_mean).item()



# Crop the image and apply mask
def crop_image_mask(image: Image.Image, mask: Image.Image):
    image_np = np.array(image)
    mask_np = np.array(mask.convert("L"))

    mask_bin = (mask_np > 0).astype(np.uint8)
    coords = np.argwhere(mask_bin)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    bbox_height = y_max - y_min + 1
    bbox_width = x_max - x_min + 1
    side = max(bbox_height, bbox_width)

    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2

    # Compute square crop bounds
    y0 = center_y - side // 2
    x0 = center_x - side // 2
    y1 = y0 + side
    x1 = x0 + side

    # Clamp to image bounds
    y0 = max(y0, 0)
    x0 = max(x0, 0)
    y1 = min(y1, image_np.shape[0])
    x1 = min(x1, image_np.shape[1])

    # Adjust if crop got smaller than 'side' due to clamping
    crop = image_np[y0:y1, x0:x1]
    mask_crop = mask_np[y0:y1, x0:x1]

    # Pad if needed to make square
    h, w = crop.shape[:2]
    pad_h = side - h
    pad_w = side - w

    if pad_h > 0 or pad_w > 0:
        crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        mask_crop = np.pad(mask_crop, ((0, pad_h), (0, pad_w)), mode='constant')

    # Apply mask
    crop_masked = crop * np.expand_dims(mask_crop, axis=-1)

    # Resize to 224x224
    result_image = Image.fromarray(crop_masked).resize((224, 224))
   
    return result_image
