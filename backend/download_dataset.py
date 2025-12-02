import os
import kagglehub
import pandas as pd
import shutil
from pathlib import Path

print("=" * 60)
print("Downloading Flickr8k Dataset from Kaggle")
print("=" * 60)

# Download latest version
print("\n📥 Downloading dataset...")
path = kagglehub.dataset_download("adityajn105/flickr8k")
print(f"✅ Dataset downloaded to: {path}")

# Setup directories
data_dir = Path("../data")
images_dir = data_dir / "images"
data_dir.mkdir(exist_ok=True)
images_dir.mkdir(exist_ok=True)

print(f"\n📁 Data directory: {data_dir.absolute()}")
print(f"📁 Images directory: {images_dir.absolute()}")

# Find the downloaded files
download_path = Path(path)
print(f"\n🔍 Exploring downloaded files in: {download_path}")

# List all files
all_files = list(download_path.rglob("*"))
print(f"\nFound {len(all_files)} files/folders")

# Find images and captions
image_files = list(download_path.rglob("*.jpg")) + list(download_path.rglob("*.png"))
caption_files = list(download_path.rglob("*.txt")) + list(download_path.rglob("*.csv"))

print(f"\n📸 Found {len(image_files)} images")
print(f"📝 Found {len(caption_files)} caption files")

# Copy images
if image_files:
    print(f"\n📋 Copying images to {images_dir}...")
    for i, img_file in enumerate(image_files[:10]):  # Show first 10
        dest = images_dir / img_file.name
        if not dest.exists():
            shutil.copy2(img_file, dest)
        if i < 5:
            print(f"  ✓ {img_file.name}")
    
    # Copy remaining silently
    for img_file in image_files[10:]:
        dest = images_dir / img_file.name
        if not dest.exists():
            shutil.copy2(img_file, dest)
    
    print(f"✅ Copied {len(image_files)} images")

# Process captions
print("\n📝 Processing captions...")

# Look for captions.txt or similar
caption_file = None
for cf in caption_files:
    if 'caption' in cf.name.lower() or 'results' in cf.name.lower():
        caption_file = cf
        print(f"  Found caption file: {cf.name}")
        break

if caption_file:
    print(f"\n📖 Reading captions from: {caption_file.name}")
    
    # Read the caption file
    with open(caption_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"  Total lines: {len(lines)}")
    
    # Parse captions (format: image_name,caption or image_name#caption_num\tcaption)
    data = []
    for line in lines[1:]:  # Skip header if exists
        line = line.strip()
        if not line:
            continue
        
        # Try different formats
        if ',' in line:
            parts = line.split(',', 1)
        elif '\t' in line:
            parts = line.split('\t', 1)
        else:
            continue
        
        if len(parts) == 2:
            image_name = parts[0].split('#')[0].strip()  # Remove #0, #1, etc.
            caption = parts[1].strip()
            
            # Only include if image exists
            if (images_dir / image_name).exists():
                data.append({
                    'image_path': f'images/{image_name}',
                    'caption': caption
                })
    
    print(f"  Parsed {len(data)} image-caption pairs")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_csv = data_dir / "flickr8k_raw.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved captions to: {output_csv}")
    print(f"   Total entries: {len(df)}")
    print(f"   Unique images: {df['image_path'].nunique()}")
    
    # Show sample
    print("\n📋 Sample entries:")
    for i in range(min(3, len(df))):
        print(f"\n  {i+1}. Image: {df.iloc[i]['image_path']}")
        print(f"     Caption: {df.iloc[i]['caption'][:80]}...")

else:
    print("⚠️  No caption file found. Checking for alternative formats...")
    
    # Try to find any CSV or text file
    for cf in caption_files:
        print(f"  Checking: {cf.name}")
        try:
            if cf.suffix == '.csv':
                df = pd.read_csv(cf)
                print(f"    Columns: {df.columns.tolist()}")
                print(f"    Rows: {len(df)}")
        except Exception as e:
            print(f"    Error: {e}")

print("\n" + "=" * 60)
print("✅ Dataset preparation complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Run: python augment_captions.py")
print("2. Run: python train.py")
print("3. Run: python app.py")
print("=" * 60)
