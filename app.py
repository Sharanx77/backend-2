from flask import Flask, request, jsonify
from utils.deepfake_detect import detect_deepfake_image
from utils.video_utils import extract_audio_emotions, extract_frames
from utils.emotion_check import check_emotional_consistency
import os

# Initialize Flask app
app = Flask(__name__)

# Root route to confirm backend is running
@app.route('/')
def home():
    return 'AI Mirror Backend is running!'

# Deepfake image detection endpoint
@app.route('/detect/image', methods=['POST'])
def detect_image():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image file provided'}), 400

    result = detect_deepfake_image(file)
    return jsonify(result)

# Video emotional consistency detection endpoint
@app.route('/detect/video', methods=['POST'])
def detect_video():
    file = request.files.get('video')
    if not file:
        return jsonify({'error': 'No video file provided'}), 400

    # Save to temp directory
    os.makedirs('temp', exist_ok=True)
    filepath = os.path.join('temp', file.filename)
    file.save(filepath)

    consistency = check_emotional_consistency(filepath)
    return jsonify({'emotional_consistency': consistency})

# Use dynamic port for Render deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

