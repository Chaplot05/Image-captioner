import os
import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import random

# Configuration
INPUT_CSV = "../data/flickr8k_raw.csv"
OUTPUT_CSV = "../data/flickr8k_augmented.csv"
MODEL_NAME = "t5-small"
BATCH_SIZE = 16
MOODS = ["funny", "sad", "romantic", "aesthetic", "motivational", "edgy", "sarcastic", "genz"]

# Limit dataset for faster training
MAX_IMAGES = 500  # Use 500 images for rapid demo training
MAX_CAPTIONS_PER_IMAGE = 1  # Use 1 caption per image

def load_model():
    print(f"Loading {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer, device

def rewrite_captions(df, model, tokenizer, device):
    new_rows = []
    
    for mood in MOODS:
        print(f"\n🎨 Generating '{mood}' captions...")
        prompts = [f"Rewrite the following caption in a {mood} style: {cap}" for cap in df['caption']]
        
        # Process in batches
        for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"{mood}"):
            batch_prompts = prompts[i:i+BATCH_SIZE]
            batch_indices = range(i, min(i+BATCH_SIZE, len(prompts)))
            
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_length=60, 
                    num_beams=5, 
                    num_return_sequences=1,
                    early_stopping=True
                )
            
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for idx, pred in zip(batch_indices, decoded_preds):
                original_row = df.iloc[idx]
                new_rows.append({
                    "image_path": original_row['image_path'],
                    "caption": pred,
                    "mood": f"<{mood}>"
                })

    return pd.DataFrame(new_rows)

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found. Please run download_dataset.py first.")
        return

    print("="*60)
    print("📊 Loading dataset...")
    df = pd.read_csv(INPUT_CSV)
    print(f"✅ Loaded {len(df):,} total captions for {df['image_path'].nunique():,} images")
    
    # Sample dataset for faster training
    if MAX_IMAGES:
        print(f"\n🎯 Sampling {MAX_IMAGES:,} images for faster training...")
        unique_images = df['image_path'].unique()
        if len(unique_images) > MAX_IMAGES:
            random.seed(42)  # For reproducibility
            sampled_images = random.sample(list(unique_images), MAX_IMAGES)
            df = df[df['image_path'].isin(sampled_images)]
            print(f"✅ Sampled {len(df):,} captions from {MAX_IMAGES:,} images")
    
    # Limit captions per image
    if MAX_CAPTIONS_PER_IMAGE:
        print(f"\n🎯 Limiting to {MAX_CAPTIONS_PER_IMAGE} captions per image...")
        df = df.groupby('image_path').head(MAX_CAPTIONS_PER_IMAGE).reset_index(drop=True)
        print(f"✅ Using {len(df):,} captions")
    
    print(f"\n📝 Final dataset: {len(df):,} captions from {df['image_path'].nunique():,} images")
    print("="*60)

    model, tokenizer, device = load_model()
    
    print(f"\n🎨 Generating {len(MOODS)} mood variations for each caption...")
    print(f"📊 Total captions to generate: {len(df) * len(MOODS):,}")
    print("="*60)
    
    augmented_df = rewrite_captions(df, model, tokenizer, device)
    
    # Add neutral mood for base captions
    df['mood'] = "<neutral>"
    
    # Combine
    final_df = pd.concat([df, augmented_df], ignore_index=True)
    
    print(f"\n💾 Saving {len(final_df):,} total captions to {OUTPUT_CSV}...")
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*60)
    print("✅ Augmentation complete!")
    print("="*60)
    print(f"📊 Statistics:")
    print(f"   - Total captions: {len(final_df):,}")
    print(f"   - Unique images: {final_df['image_path'].nunique():,}")
    print(f"   - Moods: {final_df['mood'].nunique()}")
    print(f"   - Captions per mood: ~{len(final_df) // final_df['mood'].nunique():,}")
    print("="*60)
    print("\n🚀 Next step: Run 'python train.py' to train the model!")
    print("="*60)

if __name__ == "__main__":
    main()
