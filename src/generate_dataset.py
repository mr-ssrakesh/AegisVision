import os
import sys
import random
from PIL import Image, ImageDraw, ImageFilter

# Reconfigure stdout for Windows console emoji support
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def create_dirs(base_path):
    """Create directory structure for training and validation splits."""
    for split in ['train', 'val']:
        for label in ['crack', 'rust', 'wear', 'normal']:
            os.makedirs(os.path.join(base_path, split, label), exist_ok=True)

def generate_base_texture():
    """Generate a base metallic gray texture with subtle noise."""
    img = Image.new('RGB', (224, 224), color=(180, 180, 185))
    draw = ImageDraw.Draw(img)
    
    # Add subtle metallic noise
    for _ in range(300):
        x = random.randint(0, 223)
        y = random.randint(0, 223)
        gray = random.randint(160, 200)
        draw.point((x, y), fill=(gray, gray, gray + 5))
    return img

def draw_crack(img):
    """Draw a jagged dark crack line using a random walk."""
    draw = ImageDraw.Draw(img)
    start_x = random.randint(30, 194)
    start_y = random.randint(10, 40)
    current_x, current_y = start_x, start_y
    
    for _ in range(12):
        next_x = current_x + random.randint(-15, 15)
        next_y = current_y + random.randint(12, 18)
        
        # Keep inside bounds
        next_x = max(10, min(213, next_x))
        next_y = max(10, min(213, next_y))
        
        width = random.randint(1, 3)
        color = random.randint(30, 70)  # Dark gray / black crack
        draw.line([(current_x, current_y), (next_x, next_y)], fill=(color, color, color), width=width)
        current_x, current_y = next_x, next_y
        
    return img.filter(ImageFilter.GaussianBlur(0.4))

def draw_rust(img):
    """Overlay irregular orange-brown corrosion patches."""
    draw = ImageDraw.Draw(img)
    
    # Draw soft, layered rust spots
    for _ in range(random.randint(4, 7)):
        cx = random.randint(40, 180)
        cy = random.randint(40, 180)
        r = random.randint(15, 30)
        
        for offset in range(r, 0, -2):
            # Orange-brown color distribution
            red = random.randint(170, 210)
            green = random.randint(70, 120)
            blue = random.randint(15, 35)
            draw.ellipse([cx - offset, cy - offset, cx + offset, cy + offset], fill=(red, green, blue))
            
    return img.filter(ImageFilter.GaussianBlur(1.8))

def draw_wear(img):
    """Draw metallic scratch lines indicating abrasive wear."""
    draw = ImageDraw.Draw(img)
    
    # Draw parallel-ish bright metal scratches
    for _ in range(random.randint(6, 12)):
        x1 = random.randint(10, 213)
        y1 = random.randint(10, 213)
        length = random.randint(30, 80)
        angle_dx = random.randint(20, 40)
        angle_dy = random.randint(-10, 10)
        
        x2 = max(0, min(223, x1 + length * angle_dx // 40))
        y2 = max(0, min(223, y1 + length * angle_dy // 40))
        
        color = random.randint(220, 255)  # Bright, shiny scratch lines
        draw.line([(x1, y1), (x2, y2)], fill=(color, color, color), width=1)
        
    return img.filter(ImageFilter.GaussianBlur(0.2))

def build_dataset(base_path):
    """Compile the synthetic dataset folder by folder."""
    print(f"📁 Initializing folders at {base_path}...")
    create_dirs(base_path)
    
    # Define splits configuration
    splits = {
        'train': 30,  # 30 images per class for training
        'val': 10     # 10 images per class for validation
    }
    
    for split, count in splits.items():
        print(f"Generating {count} images per class for '{split}' split...")
        for i in range(count):
            # Crack
            img_crack = draw_crack(generate_base_texture())
            img_crack.save(os.path.join(base_path, split, 'crack', f'crack_{i}.jpg'))
            
            # Rust
            img_rust = draw_rust(generate_base_texture())
            img_rust.save(os.path.join(base_path, split, 'rust', f'rust_{i}.jpg'))
            
            # Wear
            img_wear = draw_wear(generate_base_texture())
            img_wear.save(os.path.join(base_path, split, 'wear', f'wear_{i}.jpg'))
            
            # Normal
            img_normal = generate_base_texture()
            img_normal.save(os.path.join(base_path, split, 'normal', f'normal_{i}.jpg'))
            
    print("✅ Dataset generation complete.")

if __name__ == "__main__":
    build_dataset("dataset")
