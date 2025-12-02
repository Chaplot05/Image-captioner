import os
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

app = FastAPI(title="Style-Controlled Image Captioning - Simple Version")

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
device = "cuda" if torch.cuda.is_available() else "cpu"

# Mood-based prompt templates for BLIP
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

# Mood-based post-processing styles
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
    global model, processor
    print("Loading BLIP model (this may take a minute on first run)...")
    try:
        # Using BLIP - a pre-trained image captioning model
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        print(f"✅ Model loaded successfully on {device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
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
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get mood prompt
        prompt = MOOD_PROMPTS.get(mood, "a photo of")
        
        # Process image with prompt
        inputs = processor(image, text=prompt, return_tensors="pt").to(device)
        
        # Generate caption
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
        
        # Decode caption
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        
        print(f"🔍 DEBUG - Mood: {mood}")
        print(f"🔍 DEBUG - Prompt: {prompt}")
        print(f"🔍 DEBUG - Raw caption: {caption}")
        
        # Remove the prompt prefix if it appears
        caption = caption.replace(prompt, "").strip()
        
        print(f"🔍 DEBUG - Cleaned caption: {caption}")
        
        # Apply mood-specific styling
        style_func = MOOD_STYLES.get(mood, lambda x: x)
        styled_caption = style_func(caption)
        
        print(f"🔍 DEBUG - Styled caption: {styled_caption}")
        
        return {"caption": styled_caption, "mood": mood, "raw_caption": caption}
        
    except Exception as e:
        print(f"Error generating caption: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "message": "Captioning API is running (BLIP model)",
        "device": device,
        "available_moods": list(MOOD_PROMPTS.keys())
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Starting AI Caption Studio Backend (Simple Version)")
    print("="*60)
    print(f"📍 Server will run on: http://localhost:8000")
    print(f"💻 Device: {device}")
    print(f"🎨 Available moods: {len(MOOD_PROMPTS)}")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
