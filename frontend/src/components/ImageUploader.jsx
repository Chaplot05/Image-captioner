import { useState, useRef } from 'react';
import './ImageUploader.css';

function ImageUploader({ onImageSelect, selectedImage }) {
    const [preview, setPreview] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef(null);

    const handleFileChange = (file) => {
        if (file && file.type.startsWith('image/')) {
            onImageSelect(file);
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreview(reader.result);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleInputChange = (e) => {
        const file = e.target.files[0];
        handleFileChange(file);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        handleFileChange(file);
    };

    const handleClick = () => {
        fileInputRef.current?.click();
    };

    const handleRemove = (e) => {
        e.stopPropagation();
        setPreview(null);
        onImageSelect(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    return (
        <div className="image-uploader">
            <h2 className="uploader-title">Upload Your Image</h2>

            <div
                className={`upload-area ${isDragging ? 'dragging' : ''} ${preview ? 'has-image' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={handleClick}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleInputChange}
                    className="file-input"
                />

                {preview ? (
                    <div className="preview-container">
                        <img src={preview} alt="Preview" className="preview-image" />
                        <div className="preview-overlay">
                            <button className="remove-button" onClick={handleRemove}>
                                <span className="remove-icon">×</span>
                                Remove
                            </button>
                            <button className="change-button" onClick={(e) => { e.stopPropagation(); handleClick(); }}>
                                <span className="change-icon">🔄</span>
                                Change Image
                            </button>
                        </div>
                    </div>
                ) : (
                    <div className="upload-placeholder">
                        <div className="upload-icon">📸</div>
                        <h3 className="upload-text">Drop your image here</h3>
                        <p className="upload-subtext">or click to browse</p>
                        <div className="upload-formats">
                            <span className="format-badge">JPG</span>
                            <span className="format-badge">PNG</span>
                            <span className="format-badge">WEBP</span>
                        </div>
                    </div>
                )}
            </div>

            {preview && (
                <div className="image-info">
                    <div className="info-item">
                        <span className="info-icon">✓</span>
                        <span className="info-text">Image ready for captioning</span>
                    </div>
                </div>
            )}
        </div>
    );
}

export default ImageUploader;
