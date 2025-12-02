import './MoodSelector.css';

function MoodSelector({ moods, selectedMood, onMoodSelect }) {
    return (
        <div className="mood-selector">
            <h2 className="selector-title">Choose Your Vibe</h2>
            <p className="selector-subtitle">Select the mood for your caption</p>

            <div className="mood-grid">
                {moods.map((mood) => (
                    <button
                        key={mood.id}
                        className={`mood-button ${selectedMood === mood.id ? 'selected' : ''}`}
                        onClick={() => onMoodSelect(mood.id)}
                        style={{
                            '--mood-color': mood.color,
                        }}
                    >
                        <span className="mood-emoji">{mood.emoji}</span>
                        <span className="mood-label">{mood.label}</span>
                        {selectedMood === mood.id && (
                            <span className="selected-indicator">✓</span>
                        )}
                    </button>
                ))}
            </div>
        </div>
    );
}

export default MoodSelector;
