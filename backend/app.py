import os
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import VisionEncoderDecoderModel, AutoImageProcessor, GPT2TokenizerFast, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

app = FastAPI(title="Style-Controlled Image Captioning")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
processor = None
tokenizer = None
model_type = None # "custom" or "blip"
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "./clip-gpt2-caption-mood" # Path to trained model

# Mood-based prompt templates for BLIP fallback
MOOD_PROMPTS = {
    "happy": "a cheerful photo of",
    "sad": "a melancholic image of",
    "funny": "a humorous picture of",
    "romantic": "a romantic scene of",
    "motivational": "an inspiring image of",
    "aesthetic": "an artistic photo of",
    "edgy": "a bold image of",
    "sarcastic": "an ironic picture of",
    "genz": "a trendy photo of",
    "bollywood": "a dramatic scene of",
    "poetic": "a poetic image of",
    "vibey": "a moody photo of",
}

# Mood-based post-processing styles for BLIP fallback
MOOD_STYLES = {
    "happy": lambda cap: f"✨ {cap} 😊",
    "sad": lambda cap: f"💔 {cap}...",
    "funny": lambda cap: f"😂 {cap} (no cap!)",
    "romantic": lambda cap: f"💕 {cap} 💘",
    "motivational": lambda cap: f"💪 {cap} - Keep going!",
    "aesthetic": lambda cap: f"✨ {cap} 🌸",
    "edgy": lambda cap: f"🔥 {cap}",
    "sarcastic": lambda cap: f"😏 {cap} (sure...)",
    "genz": lambda cap: f"✌️ {cap} fr fr",
    "bollywood": lambda cap: f"🎬 {cap} 🎭",
    "poetic": lambda cap: f"📜 {cap}...",
    "vibey": lambda cap: f"🌊 {cap} ✨",
}

def load_model():
    global model, processor, tokenizer, model_type
    print("="*60)
    print("🚀 Starting AI Caption Studio Backend")
    print("="*60)
    
    # Try to load custom trained model first
    if os.path.exists(MODEL_PATH):
        print(f"📂 Found trained model at {MODEL_PATH}")
        try:
            print("🔧 Loading custom CLIP-GPT2 model...")
            model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)
            processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
            tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
            model_type = "custom"
            print("✅ Custom model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading custom model: {e}")
            print("⚠️ Falling back to BLIP model...")
            load_blip_model()
    else:
        print(f"⚠️ Trained model not found at {MODEL_PATH}")
        print("ℹ️ Using BLIP model as fallback (no training required)")
        load_blip_model()

def load_blip_model():
    global model, processor, model_type
    try:
        print("🔧 Loading BLIP model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        model_type = "blip"
        print("✅ BLIP model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading BLIP model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/caption")
async def generate_caption(
    file: UploadFile = File(...), 
    mood: str = Form("happy")
):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        if model_type == "custom":
            # Custom CLIP-GPT2 logic
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
            mood_token = f"<{mood}>"
            
            # Handle mood token
            if mood_token not in tokenizer.additional_special_tokens:
                 # Fallback if token not found (shouldn't happen if trained correctly)
                 pass
            
            input_ids = tokenizer(mood_token, return_tensors="pt").input_ids.to(device)
            
            output_ids = model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=input_ids,
                max_length=50,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            
            caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            print(f"DEBUG: Received mood='{mood}'")
            print(f"DEBUG: Raw caption='{caption}'")
            
            # Apply styling to custom model output as well (Hybrid Approach)
            # This ensures the mood is visible even if the model is under-trained
            style_func = MOOD_STYLES.get(mood, lambda x: x)
            styled_caption = style_func(caption)
            
            print(f"DEBUG: Styled caption='{styled_caption}'")
            
            return {"caption": styled_caption, "mood": mood, "model": "custom"}
            
        else:
            # BLIP fallback logic
            prompt = MOOD_PROMPTS.get(mood, "a photo of")
            inputs = processor(image, text=prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=5,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                )
            
            caption = processor.decode(output_ids[0], skip_special_tokens=True)
            caption = caption.replace(prompt, "").strip()
            
            # Apply styling
            style_func = MOOD_STYLES.get(mood, lambda x: x)
            styled_caption = style_func(caption)
            
            return {"caption": styled_caption, "mood": mood, "model": "blip"}
        
    except Exception as e:
        print(f"Error generating caption: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "message": f"Captioning API is running ({model_type} model)",
        "device": device
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
