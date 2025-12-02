import os
# Force PyTorch backend only (no TensorFlow)
os.environ['TRANSFORMERS_NO_TF'] = '1'

import torch
from datasets import load_dataset, Dataset
from transformers import (VisionEncoderDecoderModel, AutoImageProcessor, GPT2TokenizerFast,
                          TrainingArguments, Trainer, default_data_collator,
                          CLIPVisionModel, GPT2LMHeadModel)
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
MODEL_NAME = "clip-gpt2-caption-mood"
ENCODER_MODEL = "openai/clip-vit-base-patch32"
DECODER_MODEL = "gpt2"
DATA_FILE = "../data/flickr8k_augmented.csv"
IMAGE_DIR = "../data/" # Base directory for images if paths are relative
MAX_TARGET_LENGTH = 32
BATCH_SIZE = 4  # Balanced for speed and quality
GRADIENT_ACCUMULATION_STEPS = 2 
EPOCHS = 2 # Production training

def main():
    print("="*60)
    print("🚀 Starting Training - CLIP + GPT-2 Caption Model (PRODUCTION)")
    print("="*60)
    
    # 1. Check if augmented data exists
    if not os.path.exists(DATA_FILE):
        print(f"\n❌ Error: {DATA_FILE} not found.")
        print("Please run 'python augment_captions.py' first!")
        return

    print(f"\n📊 Loading dataset from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    
    # Memory optimization: Sample if dataset is too large for this machine
    if len(df) > 5000:
        print(f"⚠️ Dataset is large ({len(df)} rows). Sampling 5,000 for production training...")
        df = df.sample(5000, random_state=42)
        
    print(f"✅ Using {len(df):,} captions")
    print(f"   - Unique images: {df['image_path'].nunique():,}")
    print(f"   - Moods: {df['mood'].nunique()}")
    
    # 2. Split into train/val
    print("\n📊 Splitting into train/validation...")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"✅ Train: {len(train_df):,} | Val: {len(val_df):,}")
    
    # 3. Initialize Models and Tokenizer
    print("\n🔧 Loading models...")
    print(f"   - Encoder: {ENCODER_MODEL}")
    print(f"   - Decoder: {DECODER_MODEL}")
    
    # Use AutoImageProcessor for better compatibility
    image_processor = AutoImageProcessor.from_pretrained(ENCODER_MODEL)
    tokenizer = GPT2TokenizerFast.from_pretrained(DECODER_MODEL)
    
    # Add special mood tokens
    mood_tokens = ["<happy>", "<sad>", "<funny>", "<motivational>", "<romantic>", 
                   "<edgy>", "<sarcastic>", "<aesthetic>", "<genz>", "<bollywood>", 
                   "<neutral>", "<poetic>", "<vibey>"]
    tokenizer.add_special_tokens({"additional_special_tokens": mood_tokens})
    tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ Added {len(mood_tokens)} mood tokens")
    
    # Manually load encoder and decoder to ensure correct configuration
    print("   - Loading CLIP Vision Encoder...")
    encoder = CLIPVisionModel.from_pretrained(ENCODER_MODEL)
    print("   - Loading GPT-2 Decoder...")
    # We must pass add_cross_attention=True HERE so the layers are created
    decoder = GPT2LMHeadModel.from_pretrained(DECODER_MODEL, add_cross_attention=True)
    
    # Create composite model
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    
    # Configure model
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = len(tokenizer)
    
    # Resize embeddings to fit new tokens
    model.decoder.resize_token_embeddings(len(tokenizer))
    print(f"✅ Model initialized with vocab size: {len(tokenizer)}")
    
    # 4. Preprocessing
    def preprocess_function(examples):
        pixel_values = []
        labels = []
        
        for image_path, caption, mood in zip(examples['image_path'], examples['caption'], examples['mood']):
            # Handle image path
            full_image_path = os.path.join(IMAGE_DIR, image_path)
            if not os.path.exists(full_image_path):
                continue
                
            try:
                image = Image.open(full_image_path).convert("RGB")
                pixel_val = image_processor(image, return_tensors="pt").pixel_values[0]
                
                # Format: <mood> caption
                mood_str = str(mood) if pd.notna(mood) else "<neutral>"
                caption_str = str(caption)
                text = mood_str + " " + caption_str
                
                tokenized = tokenizer(text, max_length=MAX_TARGET_LENGTH, padding="max_length", truncation=True)
                lbl = tokenized.input_ids
                # Replace pad tokens with -100
                lbl = [l if l != tokenizer.pad_token_id else -100 for l in lbl]
                
                pixel_values.append(pixel_val)
                labels.append(lbl)
            except Exception as e:
                print(f"⚠️  Error processing {full_image_path}: {e}")
                continue

        return {"pixel_values": pixel_values, "labels": labels}

    # 5. Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Filter existing images
    def filter_existing(example):
        return os.path.exists(os.path.join(IMAGE_DIR, example['image_path']))
    
    train_dataset = train_dataset.filter(filter_existing)
    val_dataset = val_dataset.filter(filter_existing)
    
    print(f"✅ Filtered datasets:")
    print(f"   - Train: {len(train_dataset):,} examples")
    print(f"   - Val: {len(val_dataset):,} examples")
    
    if len(train_dataset) == 0:
        print("❌ No images found! Please check your data directory.")
        return

    print("\n🔄 Preprocessing datasets (this might take a moment)...")
    # Use keep_in_memory=False and cache_file_name to avoid RAM spikes
    train_dataset = train_dataset.map(
        preprocess_function, 
        batched=True, 
        batch_size=BATCH_SIZE, 
        remove_columns=train_dataset.column_names,
        desc="Processing Train Data"
    )
    val_dataset = val_dataset.map(
        preprocess_function, 
        batched=True, 
        batch_size=BATCH_SIZE, 
        remove_columns=val_dataset.column_names,
        desc="Processing Val Data"
    )

    print(f"✅ Preprocessing complete!")

    # 6. Training
    print("\n🏋️ Setting up training...")
    training_args = TrainingArguments(
        output_dir=f"./{MODEL_NAME}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        save_steps=200,
        eval_steps=200,
        logging_steps=50,
        eval_strategy="steps",
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
        dataloader_num_workers=0, # Avoid multiprocessing overhead on Windows
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer, # Use processing_class instead of tokenizer for newer versions
        data_collator=default_data_collator,
    )

    print("\n" + "="*60)
    print("🚀 Starting training...")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("="*60 + "\n")
    
    trainer.train()
    
    print("\n💾 Saving model...")
    trainer.save_model(f"./{MODEL_NAME}")
    tokenizer.save_pretrained(f"./{MODEL_NAME}")
    image_processor.save_pretrained(f"./{MODEL_NAME}")
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60)
    print(f"📁 Model saved to: ./{MODEL_NAME}")
    print("\n🚀 Next step: Update app.py to use the trained model!")
    print("="*60)

if __name__ == "__main__":
    main()
