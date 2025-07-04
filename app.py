from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import traceback
from fpdf import FPDF
import os

app = Flask(__name__)
CORS(app)

# A simple PDF class to handle headers and footers
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'SCORA Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

@app.route('/')
def home():
    return "SCORA backend is running"

@app.route('/analyze', methods=['POST'])
def analyze_route():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid or empty JSON payload"}), 400

        images_b64 = data.get('images', [])
        if not images_b64:
            return jsonify({"error": "No images provided"}), 400

        results = []
        for i, b64_string in enumerate(images_b64):
            try:
                score = len(b64_string) % 101 / 100 
                mock_metrics = {
                    "sharpness": score * 2000, "contrast": score * 100, "brightness": 50 + (score * 50),
                    "entropy": 5 + (score * 3), "noise": (1 - score) * 10, "dpi": 300,
                    "unique_colors": 5000 + (score * 10000), "colorfulness": 10 + (score * 50),
                    "exposure_quality": score, "rule_of_thirds": score
                }
                normalized_metrics = {k: v / max(1, mock_metrics.get(k, 1)) for k, v in mock_metrics.items()}
                results.append({
                    "index": i, "score": score * 10,
                    "tier": "Top" if score > 0.7 else "Review" if score > 0.4 else "Reject",
                    "metrics": mock_metrics, "normalized_metrics": normalized_metrics,
                    "weights": data.get('weights', {})
                })
            except Exception as e:
                results.append({"index": i, "error": str(e)})

        return jsonify({"results": results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500

@app.route('/generate-report', methods=['POST'])
def generate_report_route():
    temp_image_files = []
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid or empty JSON payload"}), 400

        selected_results = data.get('selectedResults', [])
        if not selected_results:
            return jsonify({"error": "No results selected for the report"}), 400

        pdf = PDF()
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, f"Project: {data.get('projectName', 'N/A')}", 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Client: {data.get('clientName', 'N/A')}", 0, 1)
        pdf.ln(10)

        for item in selected_results:
            try:
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, f"Filename: {item.get('filename', 'N/A')}", 0, 1)
                
                header, encoded = item['thumbnail_b64'].split(",", 1)
                image_data = base64.b64decode(encoded)
                
                image_file_path = f"temp_image_{item['index']}.png"
                temp_image_files.append(image_file_path)
                with open(image_file_path, "wb") as f:
                    f.write(image_data)
                
                pdf.image(image_file_path, x=pdf.get_x(), w=190)
                pdf.ln(105) # Adjust this value based on your image height to prevent overlap

                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 10, f"Score: {item.get('score', 0):.2f}", 0, 1)
                pdf.cell(0, 10, f"Tier: {item.get('tier', 'N/A')}", 0, 1)
                pdf.ln(10)

            except Exception as e:
                pdf.cell(0, 10, f"Could not process item {item.get('index', 'N/A')}: {e}", 0, 1)
                print(f"Error processing image for PDF: {e}")
        
        # Corrected line: removed the unnecessary .encode()
        pdf_buffer = io.BytesIO(pdf.output())
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name='scora_report.pdf',
            mimetype='application/pdf'
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Report generation server error: {str(e)}"}), 500
    finally:
        # Clean up temporary files
        for f in temp_image_files:
            if os.path.exists(f):
                os.remove(f)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))
