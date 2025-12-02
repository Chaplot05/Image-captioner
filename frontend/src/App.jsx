import { useState } from 'react'
import './App.css'
import ImageUploader from './components/ImageUploader'
import MoodSelector from './components/MoodSelector'
import CaptionDisplay from './components/CaptionDisplay'
import BackgroundEffects from './components/BackgroundEffects'

const MOODS = [
  { id: 'happy', label: 'Happy', emoji: '😄', color: 'hsl(45, 95%, 60%)' },
  { id: 'sad', label: 'Sad', emoji: '😢', color: 'hsl(220, 60%, 60%)' },
  { id: 'funny', label: 'Funny', emoji: '😂', color: 'hsl(30, 90%, 60%)' },
  { id: 'romantic', label: 'Romantic', emoji: '💘', color: 'hsl(340, 85%, 60%)' },
  { id: 'motivational', label: 'Motivational', emoji: '💪', color: 'hsl(150, 70%, 50%)' },
  { id: 'aesthetic', label: 'Aesthetic', emoji: '✨', color: 'hsl(280, 85%, 70%)' },
  { id: 'edgy', label: 'Edgy', emoji: '🔥', color: 'hsl(0, 85%, 60%)' },
  { id: 'sarcastic', label: 'Sarcastic', emoji: '😏', color: 'hsl(180, 60%, 50%)' },
  { id: 'genz', label: 'Gen Z', emoji: '🤪', color: 'hsl(300, 80%, 65%)' },
  { id: 'bollywood', label: 'Bollywood', emoji: '🎬', color: 'hsl(35, 95%, 55%)' },
  { id: 'poetic', label: 'Poetic', emoji: '📜', color: 'hsl(260, 70%, 65%)' },
  { id: 'vibey', label: 'Vibey', emoji: '🌊', color: 'hsl(200, 90%, 55%)' },
];

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedMood, setSelectedMood] = useState('happy');
  const [caption, setCaption] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState('');

  const handleImageSelect = (file) => {
    setSelectedImage(file);
    setCaption('');
    setError('');
  };

  const handleMoodSelect = (moodId) => {
    setSelectedMood(moodId);
  };

  const generateCaption = async () => {
    if (!selectedImage) {
      setError('Please upload an image first');
      return;
    }

    setIsGenerating(true);
    setError('');
    
    const formData = new FormData();
    formData.append('file', selectedImage);
    formData.append('mood', selectedMood);

    try {
      const response = await fetch('http://localhost:8000/caption', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to generate caption');
      }

      const data = await response.json();
      setCaption(data.caption);
    } catch (err) {
      setError(err.message || 'Failed to generate caption. Make sure the backend is running.');
      console.error('Error:', err);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="app">
      <BackgroundEffects />
      
      <header className="app-header">
        <div className="header-content">
          <h1 className="app-title">
            <span className="gradient-text">AI Caption Studio</span>
          </h1>
          <p className="app-subtitle">
            Generate creative, mood-based captions for your images using AI
          </p>
        </div>
      </header>

      <main className="app-main">
        <div className="container">
          <div className="content-grid">
            <div className="upload-section">
              <ImageUploader 
                onImageSelect={handleImageSelect}
                selectedImage={selectedImage}
              />
            </div>

            <div className="controls-section">
              <MoodSelector
                moods={MOODS}
                selectedMood={selectedMood}
                onMoodSelect={handleMoodSelect}
              />

              <button 
                className="generate-button"
                onClick={generateCaption}
                disabled={!selectedImage || isGenerating}
              >
                {isGenerating ? (
                  <>
                    <span className="spinner"></span>
                    Generating...
                  </>
                ) : (
                  <>
                    <span className="button-icon">✨</span>
                    Generate Caption
                  </>
                )}
              </button>

              {error && (
                <div className="error-message">
                  <span className="error-icon">⚠️</span>
                  {error}
                </div>
              )}

              <CaptionDisplay 
                caption={caption}
                mood={MOODS.find(m => m.id === selectedMood)}
              />
            </div>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>Built with ❤️ using CLIP + GPT-2 | Vision Encoder-Decoder Architecture</p>
      </footer>
    </div>
  )
}

export default App
