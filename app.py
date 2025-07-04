from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import traceback
from fpdf import FPDF

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
        # In a real app, you would use the weights and mode for analysis
        # weights = data.get('weights', {})
        # mode = data.get('mode', 'absolute')

        if not images_b64:
            return jsonify({"error": "No images provided"}), 400

        results = []
        for i, b64_string in enumerate(images_b64):
            try:
                # Mock analysis logic
                score = len(b64_string) % 101 / 100 
                
                # Mock metrics for demonstration
                mock_metrics = {
                    "sharpness": score * 2000,
                    "contrast": score * 100,
                    "brightness": 50 + (score * 50),
                    "entropy": 5 + (score * 3),
                    "noise": (1 - score) * 10,
                    "dpi": 300,
                    "unique_colors": 5000 + (score * 10000),
                    "colorfulness": 10 + (score * 50),
                    "exposure_quality": score,
                    "rule_of_thirds": score
                }
                
                normalized_metrics = {k: v / max(1, mock_metrics[k]) for k, v in mock_metrics.items()}

                results.append({
                    "index": i,
                    "score": score * 10,
                    "tier": "Top" if score > 0.7 else "Review" if score > 0.4 else "Reject",
                    "metrics": mock_metrics,
                    "normalized_metrics": normalized_metrics,
                    "weights": data.get('weights', {}) # Pass weights back
                })
            except Exception as e:
                results.append({"index": i, "error": str(e)})

        return jsonify({"results": results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500

@app.route('/generate-report', methods=['POST'])
def generate_report_route():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid or empty JSON payload"}), 400

        selected_results = data.get('selectedResults', [])
        client_name = data.get('clientName', 'N/A')
        project_name = data.get('projectName', 'N/A')

        if not selected_results:
            return jsonify({"error": "No results selected for the report"}), 400

        pdf = PDF()
        pdf.add_page()
        pdf.set_font('Arial', '', 12)

        # Report Header
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, f'Project: {project_name}', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Client: {client_name}', 0, 1)
        pdf.ln(10)

        # Add results to PDF
        for item in selected_results:
            pdf.set_font('Arial', 'B', 12)
            # The frontend sends the full data URL, so we need to decode it
            try:
                header, encoded = item['thumbnail_b64'].split(",", 1)
                image_data = base64.b64decode(encoded)
                
                # Use a unique name for the image file in memory
                image_file_path = f"temp_image_{item['index']}.png"
                with open(image_file_path, "wb") as f:
                    f.write(image_data)
                
                pdf.cell(0, 10, f"Filename: {item.get('filename', 'N/A')}", 0, 1)
                pdf.image(image_file_path, x=10, y=pdf.get_y(), w=80)
                pdf.ln(85) # Move down to avoid overlap
            except Exception as e:
                pdf.cell(0, 10, f"Could not display image for item {item.get('index', 'N/A')}", 0, 1)
                print(f"Error processing image for PDF: {e}")


            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 10, f"Score: {item.get('score', 0):.2f}", 0, 1)
            pdf.cell(0, 10, f"Tier: {item.get('tier', 'N/A')}", 0, 1)
            pdf.ln(5)

        # Create PDF in memory
        pdf_buffer = io.BytesIO()
        pdf_content = pdf.output(dest='S').encode('latin-1')
        pdf_buffer.write(pdf_content)
        pdf_buffer.seek(0)
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name='scora_report.pdf',
            mimetype='application/pdf'
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An unexpected server error occurred during report generation: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
