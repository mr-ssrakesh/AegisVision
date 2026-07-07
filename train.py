import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# Reconfigure stdout for Windows console emoji support
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def train_and_benchmark(model_name, train_loader, val_loader, train_dataset, val_dataset, device, epochs=5):
    print(f"\n==================================================")
    print(f"🎬 Starting Benchmarking for: {model_name.upper()}")
    print(f"==================================================")
    
    # 1. Load Pre-trained Model & Adjust Head
    if model_name == 'resnet18':
        try:
            from torchvision.models import ResNet18_Weights
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            model = models.resnet18(pretrained=True)
            
        # Freeze conv layers
        for param in model.parameters():
            param.requires_grad = False
            
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 4)
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
        model_save_name = 'defect_resnet.pth'
        
    elif model_name == 'efficientnet_b0':
        try:
            from torchvision.models import EfficientNet_B0_Weights
            model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        except Exception:
            model = models.efficientnet_b0(pretrained=True)
            
        # Freeze conv layers
        for param in model.parameters():
            param.requires_grad = False
            
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, 4)
        )
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
        model_save_name = 'defect_effnet_b0.pth'
        
    elif model_name == 'efficientnet_b1':
        try:
            from torchvision.models import EfficientNet_B1_Weights
            model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
        except Exception:
            model = models.efficientnet_b1(pretrained=True)
            
        # Freeze conv layers
        for param in model.parameters():
            param.requires_grad = False
            
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, 4)
        )
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
        model_save_name = 'defect_effnet_b1.pth'
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    total_params, trainable_params = count_parameters(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 2. Training Loop
    start_train_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = (correct_train / total_train) * 100
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = (correct_val / total_val) * 100
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
        
    train_time = time.perf_counter() - start_train_time
    
    # 3. Measure Inference Latency (Time per single image on CPU in ms)
    model.eval()
    latencies = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            for i in range(inputs.size(0)):
                single_img = inputs[i].unsqueeze(0)
                t0 = time.perf_counter()
                _ = model(single_img)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000) # to ms
                
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    
    # 4. Save Weights
    model_path = os.path.join('models', model_save_name)
    torch.save(model.state_dict(), model_path)
    print(f"💾 Saved {model_name} weights to {model_path}")
    
    return {
        'model_name': model_name.upper(),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'val_acc': epoch_val_acc,
        'train_time_sec': train_time,
        'avg_latency_ms': avg_latency
    }

def main():
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    
    # 1. Transforms with Advanced Data Augmentation for Real-World Generalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.25),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Data Loaders
    print("📦 Loading image datasets...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    device = torch.device('cpu') # Force CPU for local stability and latency consistency
    
    # 3. Train all models
    models_to_train = ['resnet18', 'efficientnet_b0', 'efficientnet_b1']
    results = []
    
    for m in models_to_train:
        res = train_and_benchmark(m, train_loader, val_loader, train_dataset, val_dataset, device, epochs=5)
        results.append(res)
        
    # 4. Print Comparison Table
    print("\n" + "="*95)
    print(f"{'Model Architecture':<22} | {'Total Params':<14} | {'Trainable':<12} | {'Val Acc (%)':<12} | {'Train Time (s)':<15} | {'Latency/Img (ms)':<15}")
    print("-"*95)
    for r in results:
        print(f"{r['model_name']:<22} | {r['total_params']:<14,} | {r['trainable_params']:<12,} | {r['val_acc']:<12.2f} | {r['train_time_sec']:<15.2f} | {r['avg_latency_ms']:<15.2f}")
    print("="*95 + "\n")

    # Save comparative metrics report as JSON for app loading
    import json
    metrics_path = os.path.join('models', 'model_comparison.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"📊 Saved comparison report to {metrics_path}\n")

if __name__ == '__main__':
    main()
