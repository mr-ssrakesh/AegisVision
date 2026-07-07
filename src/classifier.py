import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Reconfigure stdout for Windows console emoji support
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ImageFolder sorts classes alphabetically: 'crack' (0), 'normal' (1), 'rust' (2), 'wear' (3)
CLASS_MAPPING = {
    0: 'crack',
    1: 'normal',
    2: 'rust',
    3: 'wear'
}

def load_model(model_name):
    """Load model architecture and apply saved weights."""
    if model_name == 'resnet18':
        model = models.resnet18()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 4)
        weight_file = 'models/defect_resnet.pth'
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0()
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, 4)
        )
        weight_file = 'models/defect_effnet_b0.pth'
    elif model_name == 'efficientnet_b1':
        model = models.efficientnet_b1()
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, 4)
        )
        weight_file = 'models/defect_effnet_b1.pth'
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if os.path.exists(weight_file):
        try:
            model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            model.eval()
            print(f"✅ Loaded trained model weights from {weight_file}")
            return model, True
        except Exception as e:
            print(f"⚠️ Error loading weights for {model_name}: {e}")
    else:
        print(f"⚠️ Weights file not found: {weight_file}")
    
    return None, False

def rule_based_fallback(img):
    """Rule-based visual feature classification if PyTorch weights are missing."""
    img_rgb = img.convert('RGB')
    width, height = img_rgb.size
    pixels = list(img_rgb.getdata())
    
    # 1. Rust Check (Color hue analysis: high Red and low Blue)
    rust_pixels = 0
    total_pixels = len(pixels)
    for r, g, b in pixels:
        if r > 130 and g > 50 and g < 140 and b < 50:
            rust_pixels += 1
    rust_ratio = rust_pixels / total_pixels
    if rust_ratio > 0.02:
        return 'rust', min(0.65 + rust_ratio, 0.98), 'Medium'
        
    # 2. Crack Check (Edge density analysis: looking for dark, continuous lines)
    img_gray = img.convert('L')
    edge_pixels = 0
    for x in range(5, width - 5):
        for y in range(5, height - 5):
            val = img_gray.getpixel((x, y))
            if val < 80:
                edge_pixels += 1
    edge_ratio = edge_pixels / total_pixels
    if edge_ratio > 0.005 and edge_ratio < 0.05:
        return 'crack', min(0.70 + edge_ratio * 2, 0.95), 'High'
        
    # 3. Wear Check (Metallic textures with light scratches)
    wear_pixels = 0
    for r, g, b in pixels:
        if r > 210 and g > 210 and b > 210:
            wear_pixels += 1
    wear_ratio = wear_pixels / total_pixels
    if wear_ratio > 0.02:
        return 'wear', min(0.60 + wear_ratio, 0.92), 'Medium'
        
    # 4. Default to normal
    return 'normal', 0.95, 'Low'

def predict_defect(image_path_or_file) -> dict:
    """Classify defect from image path or upload file trying multiple models sequentially."""
    try:
        if isinstance(image_path_or_file, str):
            img = Image.open(image_path_or_file)
        else:
            img = Image.open(image_path_or_file)
            
        img_rgb = img.convert('RGB')
        
        # Sequentially try to load: EfficientNet-B1 -> EfficientNet-B0 -> ResNet18
        model = None
        model_loaded = False
        loaded_model_name = ""
        
        for m_name in ['efficientnet_b1', 'efficientnet_b0', 'resnet18']:
            model, model_loaded = load_model(m_name)
            if model_loaded:
                loaded_model_name = m_name.replace('_', '-').upper()
                break
                
        if model_loaded:
            eval_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = eval_transform(img_rgb).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, dim=0)
                
            pred_class = CLASS_MAPPING[predicted_idx.item()]
            conf_val = confidence.item()
            
            # Hybrid AI Guardrail
            pixels = list(img_rgb.getdata())
            rust_pixels = 0
            for r, g, b in pixels:
                if r > 120 and g > 50 and g < 140 and b < 50:
                    rust_pixels += 1
            rust_ratio = rust_pixels / len(pixels)
            
            if pred_class != 'rust' and rust_ratio > 0.03:
                pred_class = 'rust'
                conf_val = max(conf_val, 0.70 + rust_ratio)
                method = f"Hybrid (DL {loaded_model_name} + Color Guardrail)"
            else:
                method = f"Deep Learning ({loaded_model_name})"
            
            severity_map = {
                'crack': 'High',
                'rust': 'Medium',
                'wear': 'Medium',
                'normal': 'Low'
            }
            
            return {
                "class": pred_class,
                "confidence": conf_val,
                "severity": severity_map[pred_class],
                "method": method
            }
        else:
            # Fallback to rule-based visual analyzer
            pred_class, conf_val, severity_val = rule_based_fallback(img_rgb)
            return {
                "class": pred_class,
                "confidence": conf_val,
                "severity": severity_val,
                "method": "Visual Feature Heuristics"
            }
            
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {
            "class": "normal",
            "confidence": 0.50,
            "severity": "Low",
            "method": "Error Fallback"
        }
