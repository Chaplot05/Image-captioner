import pandas as pd
import os

print("Verifying dataset...")
df = pd.read_csv('../data/flickr8k_raw.csv')
print(f'✅ Total rows: {len(df)}')
print(f'✅ Unique images: {df["image_path"].nunique()}')
print(f'\n📋 Sample data:')
print(df.head(3))

# Check if images exist
sample_image = df.iloc[0]['image_path']
full_path = os.path.join('../data', sample_image)
if os.path.exists(full_path):
    print(f'\n✅ Sample image exists: {sample_image}')
else:
    print(f'\n❌ Sample image NOT found: {full_path}')
