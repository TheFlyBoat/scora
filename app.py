
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import traceback

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "SCORA backend is running"

@app.route('/analyze', methods=['POST'])
def analyze_route():
    try:
        data = request.get_json(force=True)
        images = data.get('images', [])
        weights = data.get('weights', {})
        mode = data.get('mode', 'absolute')

        if not images:
            return jsonify({"error": "No images provided"}), 400

        results = []

        for i, base64_string in enumerate(images):
            try:
                if "," not in base64_string:
                    raise ValueError("Invalid base64 format")

                header, encoded = base64_string.split(",", 1)
                image_data = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(image_data)).convert("RGB")

                # Example score logic: length of data as mock score
                score = len(image_data) % 100 / 100
                results.append({
                    "index": i,
                    "score": round(score, 3),
                    "tier": "Top" if score > 0.7 else "Review" if score > 0.5 else "Reject"
                })

            except Exception as img_err:
                print(f"Image {i} failed to process:")
                traceback.print_exc()
                results.append({
                    "index": i,
                    "error": f"Invalid image: {str(img_err)}"
                })

        return jsonify({"results": results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500
