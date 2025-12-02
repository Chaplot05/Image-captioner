# AI Caption Studio 🎨✨

A modern, AI-powered image captioning application that generates creative, mood-based captions for your images. Built with CLIP Vision Encoder + GPT-2 Decoder architecture for style-controlled text generation.

## 🌟 Features

- **12 Mood Styles**: Happy, Sad, Funny, Romantic, Motivational, Aesthetic, Edgy, Sarcastic, Gen Z, Bollywood, Poetic, Vibey
- **Modern Architecture**: Vision Encoder-Decoder model (CLIP + GPT-2)
- **Beautiful UI**: Glassmorphism, gradients, smooth animations
- **Drag & Drop**: Easy image upload with preview
- **One-Click Copy**: Copy captions instantly
- **Responsive Design**: Works on desktop, tablet, and mobile

## 🏗️ Architecture

```
Vision Encoder (CLIP ViT-B/32) → Image Embeddings
                                        ↓
Transformer Decoder (GPT-2) + Mood Token → Stylized Caption
```

### How It Works

1. **Image Encoding**: CLIP encodes the image into embeddings
2. **Mood Control**: Special tokens like `<romantic>`, `<funny>` prepended to decoder
3. **Caption Generation**: GPT-2 generates captions conditioned on both image and mood
4. **Style Transfer**: Model learns to match mood through augmented training data

## 📁 Project Structure

```
Captioner/
├── backend/
│   ├── app.py                 # FastAPI server
│   ├── train.py               # Training script
│   ├── augment_captions.py    # Dataset augmentation
│   ├── create_dummy_data.py   # Test data generator
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── App.jsx           # Main app
│   │   └── index.css         # Design system
│   └── package.json
└── data/
    ├── flickr8k_raw.csv      # Base dataset
    ├── flickr8k_augmented.csv # Augmented dataset
    └── images/               # Image files
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- CUDA-capable GPU (recommended, but CPU works)
- 8GB+ RAM

### 1. Backend Setup

```powershell
cd backend

# Install dependencies
pip install -r requirements.txt

# Create dummy data (for testing)
python create_dummy_data.py

# Augment dataset (generates mood-specific captions)
python augment_captions.py

# Train the model (this will take time!)
python train.py

# Start the API server
python app.py
```

The backend will run on `http://localhost:8000`

### 2. Frontend Setup

```powershell
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will run on `http://localhost:5173`

## 📊 Dataset Preparation

### Option 1: Use Flickr8k (Recommended)

1. Download Flickr8k dataset
2. Place images in `data/images/`
3. Create `data/flickr8k_raw.csv` with columns: `image_path`, `caption`

Example CSV:
```csv
image_path,caption
images/1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up stairs
images/1001773457_577c3a7d70.jpg,A black dog and a spotted dog are fighting
```

### Option 2: Use Your Own Dataset

1. Organize images in `data/images/`
2. Create CSV with image paths and captions
3. Run augmentation script to generate mood variations

## 🎯 Training Details

### Hyperparameters

- **Encoder**: CLIP ViT-B/32 (frozen initially)
- **Decoder**: GPT-2 (small)
- **Batch Size**: 8-32 (adjust for your GPU)
- **Epochs**: 5-10
- **Learning Rate**: 5e-5
- **Max Caption Length**: 32 tokens

### Data Augmentation

The `augment_captions.py` script uses T5 to rewrite base captions into different moods:

```
Base: "Two kids playing in the park"

<funny> → "Childhood = free therapy nobody asked for"
<romantic> → "Even the wind pauses when laughter fills the air"
<aesthetic> → "Soft sunlight, soft hearts, soft memories"
```

## 🎨 UI Features

### Design System

- **Color Palette**: Vibrant HSL colors with dark mode
- **Typography**: Inter + Outfit fonts
- **Animations**: Smooth transitions, floating elements, gradients
- **Effects**: Glassmorphism, blur, glowing shadows

### Components

1. **ImageUploader**: Drag-drop with preview
2. **MoodSelector**: 12 mood buttons with emojis
3. **CaptionDisplay**: Copy, share, save actions
4. **BackgroundEffects**: Animated gradient orbs

## 🔧 API Endpoints

### POST `/caption`

Generate a caption for an image.

**Request:**
```
FormData:
  - file: Image file
  - mood: Mood ID (e.g., "romantic", "funny")
```

**Response:**
```json
{
  "caption": "Generated caption text",
  "mood": "romantic"
}
```

### GET `/`

Health check endpoint.

## 📈 Evaluation

### Automated Metrics

- **BLEU/ROUGE**: Caption fidelity
- **CIDEr/SPICE**: Semantic relevance
- **Distinct-1/2**: Vocabulary diversity

### Human Evaluation

Rate on 3 axes (1-5 scale):
1. **Relevance**: Does it match the image?
2. **Style Match**: Does it fit the mood?
3. **Creativity**: Would you use it?

## 🚀 Deployment

### Option 1: Local Server

```powershell
# Backend
cd backend
python app.py

# Frontend (production build)
cd frontend
npm run build
npm run preview
```

### Option 2: Cloud Deployment

**Backend** (AWS/GCP/Azure):
- Use GPU instance (e.g., AWS g4dn.xlarge)
- Docker container with FastAPI
- Load model on startup

**Frontend** (Vercel/Netlify):
- Build static files: `npm run build`
- Deploy `dist/` folder
- Update API URL in `App.jsx`

### Optimization

- **Quantization**: Use 8-bit/4-bit for smaller models
- **ONNX Export**: Faster inference
- **Caching**: Cache frequent requests
- **Batching**: Process multiple images together

## 🎓 How to Improve

1. **Better Dataset**: Use COCO Captions (120k images)
2. **Larger Model**: GPT-2 Medium/Large
3. **Fine-tune Encoder**: Unfreeze CLIP after initial training
4. **More Moods**: Add custom mood tokens
5. **Beam Search**: Increase beam width for diversity
6. **Reinforcement Learning**: Use RLHF for better style matching

## 🐛 Troubleshooting

### Model Not Loading

- Check if `clip-gpt2-caption-mood/` exists in backend
- Verify all model files are present
- Try loading base models first (will work but without mood control)

### Out of Memory

- Reduce batch size in `train.py`
- Use gradient accumulation
- Use smaller model (DistilGPT2)

### Poor Caption Quality

- Train for more epochs
- Increase dataset size
- Improve augmentation quality
- Add more training examples for specific moods

### Backend Connection Failed

- Ensure backend is running on port 8000
- Check CORS settings in `app.py`
- Verify firewall/antivirus isn't blocking

## 📝 License

MIT License - Feel free to use for personal or commercial projects

## 🙏 Credits

- **CLIP**: OpenAI
- **GPT-2**: OpenAI
- **Transformers**: Hugging Face
- **Flickr8k**: University of Illinois

## 💡 Future Ideas

- [ ] Multi-language support
- [ ] Video caption generation
- [ ] Hashtag suggestions
- [ ] Instagram integration
- [ ] Caption history/favorites
- [ ] A/B testing for captions
- [ ] Custom mood creation
- [ ] Batch processing


