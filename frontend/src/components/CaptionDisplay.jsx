import { useState } from 'react';
import './CaptionDisplay.css';

function CaptionDisplay({ caption, mood }) {
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        if (caption) {
            await navigator.clipboard.writeText(caption);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    if (!caption) {
        return (
            <div className="caption-display empty">
                <div className="empty-state">
                    <span className="empty-icon">💭</span>
                    <p className="empty-text">Your caption will appear here</p>
                </div>
            </div>
        );
    }

    return (
        <div className="caption-display">
            <div className="caption-header">
                <div className="caption-mood">
                    <span className="mood-emoji-small">{mood?.emoji}</span>
                    <span className="mood-name">{mood?.label} Vibe</span>
                </div>
                <button
                    className={`copy-button ${copied ? 'copied' : ''}`}
                    onClick={handleCopy}
                >
                    {copied ? (
                        <>
                            <span className="copy-icon">✓</span>
                            Copied!
                        </>
                    ) : (
                        <>
                            <span className="copy-icon">📋</span>
                            Copy
                        </>
                    )}
                </button>
            </div>

            <div className="caption-content">
                <p className="caption-text">{caption}</p>
            </div>

            <div className="caption-actions">
                <button className="action-button share-button">
                    <span className="action-icon">📤</span>
                    Share
                </button>
                <button className="action-button save-button">
                    <span className="action-icon">💾</span>
                    Save
                </button>
                <button className="action-button regenerate-button">
                    <span className="action-icon">🔄</span>
                    Regenerate
                </button>
            </div>
        </div>
    );
}

export default CaptionDisplay;
