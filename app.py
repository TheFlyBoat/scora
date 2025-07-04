from flask import Flask, render_template, request, send_file
from flask_cors import CORS
from PIL import Image, ImageStat
import numpy as np
import cv2
import base64
from io import BytesIO
from fpdf import FPDF, XPos, YPos
import os
import json
from datetime import datetime
import re

# IMPORTANT: Set the matplotlib backend to 'Agg' before importing pyplot
# This prevents GUI-related errors when running in a non-interactive environment (like a Flask server)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

# Turn off interactive plotting to prevent GUI issues in a server environment
plt.ioff()

def make_json_serializable(obj):
    import collections.abc
    try:
        from PIL.TiffImagePlugin import IFDRational
    except ImportError:
        IFDRational = None
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif IFDRational and isinstance(obj, IFDRational):
        return float(obj)
    elif hasattr(obj, '__float__'):
        try:
            return float(obj)
        except Exception:
            return str(obj)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)


app = Flask(__name__, template_folder='.')
CORS(app)

LOGO_PATH = "scora_logo.png" 
FAVICON_PATH = "favicon.png"

# --- Absolute Scoring Targets ---
ABSOLUTE_TARGETS = {
    "sharpness": 1500.0, "contrast": 60.0, "entropy": 7.5,
    "brightness": 128.0, "dpi": 300, "rule_of_thirds": 1.0,
    "exposure_quality": 1.0, "noise": 0.0, "colorfulness": 60.0
}

# --- Image Analysis Functions ---
def get_image_properties(image_pil):
    dpi = image_pil.info.get('dpi', (72, 72)); return {"dpi": dpi[0], "width": image_pil.width, "height": image_pil.height}
def get_sharpness(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY); return cv2.Laplacian(gray, cv2.CV_64F).var()
def get_contrast(image_pil):
    gray = image_pil.convert('L'); return ImageStat.Stat(gray).stddev[0]
def get_entropy(image_pil):
    gray = image_pil.convert('L'); hist = np.array(gray.histogram()); hist_norm = hist[hist > 0] / hist.sum(); return -np.sum(hist_norm * np.log2(hist_norm))
def get_brightness(image_pil):
    gray = image_pil.convert('L'); return ImageStat.Stat(gray).mean[0]
def get_unique_colors(image_pil):
    thumbnail = image_pil.copy(); thumbnail.thumbnail((100, 100)); return len(thumbnail.getcolors(maxcolors=10000) or [])
def get_exposure_quality(image_pil):
    gray = image_pil.convert('L'); hist = gray.histogram(); total_pixels = image_pil.width * image_pil.height
    crushed_blacks = sum(hist[:5]); blown_whites = sum(hist[250:])
    bad_pixels = (crushed_blacks + blown_whites) / total_pixels; return 1.0 - bad_pixels
def get_rule_of_thirds_score(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY); height, width = gray.shape
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=10)
    if corners is None or corners.size == 0: return 0.0
    third_x, third_y = width / 3, height / 3
    intersections = [(third_x, third_y), (2 * third_x, third_y), (third_x, 2 * third_y), (2 * third_x, 2 * third_y)]
    threshold = 0.1 * min(width, height) 
    for corner in corners.reshape(-1, 2):
        for ix, iy in intersections:
            if np.sqrt((corner[0] - ix)**2 + (corner[1] - iy)**2) < threshold: return 1.0
    return 0.0
def get_noise_level(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    noise = np.sum(np.abs(gray.astype('float') - denoised.astype('float'))) / gray.size; return noise
def get_colorfulness(image_cv):
    (B, G, R) = cv2.split(image_cv.astype("float")); rg = np.absolute(R - G); yb = np.absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg)); (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2)); meanRoot = np.sqrt((ybMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)

def analyze_image(image_file):
    try:
        image_bytes = image_file.read()
        image_pil = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_cv = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        props = get_image_properties(image_pil)
        metrics = {
            "sharpness": get_sharpness(image_cv), "contrast": get_contrast(image_pil),
            "entropy": get_entropy(image_pil), "brightness": get_brightness(image_pil),
            "unique_colors": get_unique_colors(image_pil), "dpi": props["dpi"],
            "width": props["width"], "height": props["height"],
            "exposure_quality": get_exposure_quality(image_pil),
            "rule_of_thirds": get_rule_of_thirds_score(image_cv),
            "noise": get_noise_level(image_cv), "colorfulness": get_colorfulness(image_cv)
        }
        return metrics, image_pil
    except Exception as e:
        return {"error": str(e)}, None

def normalize_relative(metrics_list, active_keys):
    if not metrics_list: return []
    normalized_list = []
    transposed = {key: [d.get(key, 0) for d in metrics_list] for key in active_keys}
    min_max = {key: (min(values), max(values)) for key, values in transposed.items()}
    for metrics in metrics_list:
        normalized = {}
        for key in active_keys:
            min_v, max_v = min_max[key]; value = metrics.get(key, 0)
            if max_v - min_v == 0: normalized[key] = 0.5
            else:
                if key == 'noise': normalized[key] = 1 - ((value - min_v) / (max_v - min_v))
                else: normalized[key] = (value - min_v) / (max_v - min_v)
        normalized_list.append(normalized)
    return normalized_list

def normalize_absolute(metrics_list, active_keys):
    normalized_list = []
    RANGES = {"brightness": 255, "entropy": 8, "contrast": 128, "sharpness": 3000, "dpi": 300, "noise": 50, "colorfulness": 150}
    for metrics in metrics_list:
        normalized = {}
        for key in active_keys:
            value = metrics.get(key, 0); target = ABSOLUTE_TARGETS.get(key)
            if target is not None:
                if key in ["rule_of_thirds", "exposure_quality"]: normalized[key] = value
                else:
                    range_val = RANGES.get(key, target * 2); distance = abs(value - target)
                    normalized[key] = max(0, 1 - (distance / range_val))
            else: normalized[key] = 0.5
        normalized_list.append(normalized)
    return normalized_list

def calculate_final_score(normalized_metrics, weights):
    active_weights = {k: v for k, v in weights.items() if v > 0}
    if not active_weights: return 0
    score = sum(normalized_metrics.get(key, 0) * weight for key, weight in active_weights.items())
    total_weight = sum(active_weights.values())
    if total_weight == 0: return 0
    return (score / total_weight) * 10

def get_tier(score):
    if score >= 8: return "Top"
    elif score >= 5: return "Review"
    else: return "Reject"

def create_thumbnail(image_pil, size=(200, 200)):
    image_pil.thumbnail(size); buffered = BytesIO()
    image_pil.save(buffered, format="PNG"); return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

def generate_radar_chart_image(normalized_metrics, weights, criteria_map):
    active_criteria = {key: label for group in criteria_map.values() for key, label in group['criteria'].items()}
    labels = [active_criteria.get(key, key.replace('_', ' ').title()) for key, w in weights.items() if w > 0 and key in normalized_metrics]
    values = [normalized_metrics[key] for key, w in weights.items() if w > 0 and key in normalized_metrics]
    if not labels: return None
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='indigo', alpha=0.25)
    ax.plot(angles, values, color='indigo', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='gray', size=8)
    ax.set_ylim(0, 1)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close(fig)
    buf.seek(0)
    
    # Create a unique but simple temporary filename
    temp_chart_path = f"temp_chart_{datetime.now().timestamp()}.png"
    with open(temp_chart_path, "wb") as f:
        f.write(buf.getvalue())
    return temp_chart_path

class PDF(FPDF):
    def header(self):
        if os.path.exists(LOGO_PATH): self.image(LOGO_PATH, 10, 8, 30)
        self.set_font('Helvetica', 'B', 20)
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'SCORA Report | {datetime.now().strftime("%Y-%m-%d")} | Page {self.page_no()}', align='C')

    def title_page(self, client, project, notes):
        self.add_page()
        self.set_font('Helvetica', 'B', 28)
        self.cell(0, 20, 'Image Analysis Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(10)
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 15, 'Structured Criteria for Objective Rating of Artworks', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(20)
        self.set_font('Helvetica', 'B', 12)
        self.cell(40, 10, 'Client:')
        self.set_font('Helvetica', '', 12)
        self.cell(0, 10, client, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font('Helvetica', 'B', 12)
        self.cell(40, 10, 'Project:')
        self.set_font('Helvetica', '', 12)
        self.cell(0, 10, project, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font('Helvetica', 'B', 12)
        self.cell(40, 10, 'Date:')
        self.set_font('Helvetica', '', 12)
        self.cell(0, 10, datetime.now().strftime("%B %d, %Y"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if notes:
            self.ln(10)
            self.set_font('Helvetica', 'B', 14)
            self.cell(0, 10, 'Notes & Observations', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            self.set_font('Helvetica', '', 10)
            self.multi_cell(0, 5, notes)

    def summary_page(self, results, weights):
        self.add_page()
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Analysis Summary', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(5)
        # Summary Table
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(220, 220, 220)
        self.cell(100, 8, 'Filename', 1, 0, 'C', 1)
        self.cell(45, 8, 'Score', 1, 0, 'C', 1)
        self.cell(45, 8, 'Tier', 1, 1, 'C', 1)
        self.set_font('Helvetica', '', 9)
        for item in results:
            self.cell(100, 7, item['filename'][:55], 1)
            self.cell(45, 7, f"{item['score']:.2f}", 1, 0, 'C')
            self.cell(45, 7, item['tier'], 1, 1, 'C')
        self.ln(10)
        # Weights Table
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Active Criteria & Weights:', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.set_font('Helvetica', 'B', 10)
        self.cell(50, 8, 'Criterion', 1, 0, 'C', 1)
        self.cell(40, 8, 'Weight', 1, 1, 'C', 1)
        self.set_font('Helvetica', '', 9)
        for key, value in weights.items():
            if value > 0:
                self.cell(50, 7, key.replace('_', ' ').title(), 1)
                self.cell(40, 7, f"{value:.2f}", 1, 1, 'R')

    def image_details(self, item, options, criteria_map):
        # Check if we need to add a new page
        required_height = 150 # Estimated height for one analysis block
        if options.get('onePerPage') or (self.get_y() + required_height > self.page_break_trigger):
            self.add_page()

        tier_colors = {'Top': (230, 255, 230), 'Review': (255, 255, 230), 'Reject': (255, 230, 230)}
        self.set_fill_color(*tier_colors.get(item['tier'], (255, 255, 255)))
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, f"{item['filename']}", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L', fill=True)
        
        start_y = self.get_y()
        
        # Add thumbnail
        temp_img_path = None
        try:
            img_data = base64.b64decode(item['thumbnail'].split(',')[1])
            temp_img_path = f"temp_thumb_{item['filename'].replace('.', '_').replace('/', '_')}.png"
            with open(temp_img_path, "wb") as f: f.write(img_data)
            self.image(temp_img_path, x=15, y=start_y + 5, w=80)
        except Exception as e:
            print(f"Could not embed thumbnail for {item['filename']}: {e}")
        
        # Add radar chart
        chart_path = None
        if options.get('includeCharts'):
            try:
                chart_path = generate_radar_chart_image(item['normalized_metrics'], item['weights'], criteria_map)
                if chart_path:
                    self.image(chart_path, x=115, y=start_y + 5, w=80)
            except Exception as e:
                print(f"Could not generate or embed radar chart for {item['filename']}: {e}")

        self.set_y(start_y + 95) # Position below images
        self.set_font('Helvetica', 'B', 16)
        self.cell(90, 15, f"Overall Score: {item['score']:.2f}", align='C')
        self.set_font('Helvetica', 'B', 16)
        self.cell(90, 15, f"Tier: {item['tier']}", align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)

        # Display individual metric bars
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(220, 220, 220)
        self.cell(60, 8, 'Metric', 1, 0, 'C', 1)
        self.cell(100, 8, 'Normalized Score', 1, 0, 'C', 1)
        self.cell(30, 8, 'Raw Value', 1, 1, 'C', 1)
        self.set_font('Helvetica', '', 9)

        for key, norm_value in sorted(item['normalized_metrics'].items()):
            if item['weights'].get(key, 0) == 0: continue
            display_label = key.replace('_', ' ').title()
            for group in criteria_map.values():
                if key in group['criteria']: display_label = group['criteria'][key]
            
            self.cell(60, 8, display_label, 1)
            bar_x = self.get_x()
            self.set_fill_color(240, 240, 240)
            self.cell(100, 8, '', 1, 0, 'L', True)
            self.set_fill_color(79, 70, 229)
            if norm_value > 0:
                self.set_xy(bar_x, self.get_y())
                self.cell(100 * norm_value, 8, '', 0, 0, 'L', True)
            
            self.set_xy(bar_x + 100, self.get_y())
            self.cell(30, 8, f"{item['metrics'][key]:.1f}", 1, 1, 'C')
        
        self.ln(10) # Add space after each item

        # Clean up temporary files
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        if chart_path and os.path.exists(chart_path):
            os.remove(chart_path)

def generate_pdf_report(report_data):
    pdf = PDF()
    options = report_data.get('options', {})
    
    pdf.title_page(report_data['clientName'], report_data['projectName'], report_data.get('notes', ''))
    
    if options.get('includeSummary'):
        pdf.summary_page(report_data['selectedResults'], report_data['weights'])

    CRITERIA_GROUPS_BACKEND = {
        "Technical": { "criteria": { "sharpness": "Sharpness", "entropy": "Detail/Entropy", "noise": "Noise Level", "dpi": "Resolution (DPI)" }},
        "Color": { "criteria": { "contrast": "Contrast", "brightness": "Brightness", "unique_colors": "Color Variety", "colorfulness": "Colorfulness" }},
        "Composition": { "criteria": { "exposure_quality": "Exposure Quality", "rule_of_thirds": "Rule of Thirds" }}
    }
    
    for item in report_data['selectedResults']:
        pdf.image_details(item, options, CRITERIA_GROUPS_BACKEND)

    return pdf.output()

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

@app.route('/')
def home(): return send_file("index.html")

@app.route('/scora_logo.png')
def serve_logo(): return send_file(LOGO_PATH)

@app.route('/favicon.png')
def serve_favicon(): return send_file(FAVICON_PATH)

@app.route('/analyze', methods=['POST'])
def analyze_route():
    try:
        files = request.files.getlist('images') if 'images' in request.files else []
        if not files: return jsonify({"error": "No images provided"}), 400
        weights = json.loads(request.form.get('weights'))
        scoring_mode = request.form.get('scoring_mode', 'relative')
        active_keys = [key for key, weight in weights.items() if weight > 0]
        
        all_metrics, pil_images, filenames, errors = [], [], [], []
        for f in files:
            metrics, pil_img = analyze_image(f)
            if "error" in metrics:
                errors.append(f"{f.filename}: {metrics['error']}")
            elif metrics and pil_img: 
                all_metrics.append(metrics); pil_images.append(pil_img); filenames.append(f.filename)

        if not all_metrics and errors:
            return jsonify({"error": f"All images failed to process. First error: {errors[0]}"}), 500
        
        if scoring_mode == 'relative':
            normalized_metrics_list = normalize_relative(all_metrics, active_keys)
        else:
            normalized_metrics_list = normalize_absolute(all_metrics, active_keys)
            
        results = []
        for i, norm_metrics in enumerate(normalized_metrics_list):
            score = calculate_final_score(norm_metrics, weights)
            results.append({
                "filename": filenames[i], "metrics": all_metrics[i],
                "normalized_metrics": norm_metrics, "weights": weights,
                "score": score, "tier": get_tier(score),
                "thumbnail": create_thumbnail(pil_images[i])
            })
        results.sort(key=lambda x: x['score'], reverse=True)
        return jsonify({"results": make_json_serializable(results), "errors": make_json_serializable(errors)})
    except Exception as e:
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500

@app.route('/generate-report', methods=['POST'])
def generate_report_route():
    data = request.get_json()
    if not data or 'selectedResults' not in data: return "Invalid data", 400
    
    project_name = sanitize_filename(data.get('projectName', 'Report'))
    date_str = datetime.now().strftime("%Y-%m-%d")
    download_name = f"SCORA_{project_name}_{date_str}.pdf"

    pdf_data = generate_pdf_report(data)
    return send_file(BytesIO(pdf_data), as_attachment=True, download_name=download_name, mimetype='application/pdf')

if __name__ == "__main__":
    app.run(debug=False)

