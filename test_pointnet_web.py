#!/usr/bin/env python3
"""
Web interface for testing PointNet++ segmentation.

Run this script and open http://localhost:5001 in your browser.
Upload a mesh file (STL, OBJ, PLY) to see PointNet++ segmentation results.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from flask import Flask, render_template_string, request, jsonify, send_file
from werkzeug.utils import secure_filename
import torch
import numpy as np
import trimesh

from vlm_cad.segmentation import create_segmentation_backend
from vlm_cad.pointnet_seg.labels import get_category_from_flat_label
from vlm_cad.pointnet_seg.geometry import compute_part_statistics, compute_part_bounding_boxes

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp(prefix='pointnet_web_')

# Global model variable
_model = None
_device = None

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>PointNet++ Segmentation Test</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .upload-area {
            border: 3px dashed #ccc;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            background: #fafafa;
            transition: all 0.3s;
        }
        .upload-area:hover {
            border-color: #4CAF50;
            background: #f0f8f0;
        }
        .upload-area.dragover {
            border-color: #4CAF50;
            background: #e8f5e9;
        }
        input[type="file"] {
            margin: 10px;
            padding: 10px;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
            margin: 10px;
        }
        button:hover {
            background: #45a049;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        #status {
            margin: 20px 0;
            padding: 15px;
            border-radius: 5px;
            display: none;
        }
        #status.info {
            background: #e3f2fd;
            color: #1976d2;
        }
        #status.success {
            background: #e8f5e9;
            color: #2e7d32;
        }
        #status.error {
            background: #ffebee;
            color: #c62828;
        }
        #results {
            margin-top: 30px;
            display: none;
        }
        .part-item {
            background: #f9f9f9;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .part-item h3 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .part-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .stat {
            background: white;
            padding: 10px;
            border-radius: 5px;
        }
        .stat-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }
        .stat-value {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #4CAF50;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .download-link {
            display: inline-block;
            margin-top: 20px;
            padding: 10px 20px;
            background: #2196F3;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }
        .download-link:hover {
            background: #1976D2;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 PointNet++ Segmentation Test</h1>
        <p>Upload a 3D mesh file (STL, OBJ, PLY) to see PointNet++ part segmentation results.</p>
        
        <div class="upload-area" id="uploadArea">
            <p>📁 Drag and drop a mesh file here, or click to browse</p>
            <input type="file" id="fileInput" accept=".stl,.obj,.ply,.STL,.OBJ,.PLY" style="display: none;">
            <button onclick="document.getElementById('fileInput').click()">Choose File</button>
            <p id="fileName" style="margin-top: 10px; color: #666;"></p>
        </div>
        
        <div id="status"></div>
        
        <button id="analyzeBtn" onclick="analyzeMesh()" disabled>Analyze Mesh</button>
        
        <div id="results"></div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const statusDiv = document.getElementById('status');
        const resultsDiv = document.getElementById('results');
        const fileNameDiv = document.getElementById('fileName');
        
        let selectedFile = null;

        // Drag and drop handlers
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelect(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        function handleFileSelect(file) {
            const validExtensions = ['.stl', '.obj', '.ply'];
            const ext = '.' + file.name.split('.').pop().toLowerCase();
            
            if (!validExtensions.includes(ext)) {
                showStatus('Please select a valid mesh file (.stl, .obj, or .ply)', 'error');
                return;
            }
            
            selectedFile = file;
            fileNameDiv.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
            analyzeBtn.disabled = false;
            resultsDiv.style.display = 'none';
        }

        function showStatus(message, type = 'info') {
            statusDiv.textContent = message;
            statusDiv.className = type;
            statusDiv.style.display = 'block';
        }

        async function analyzeMesh() {
            if (!selectedFile) {
                showStatus('Please select a file first', 'error');
                return;
            }

            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<span class="loading"></span> Analyzing...';
            showStatus('Uploading and analyzing mesh... This may take 10-30 seconds.', 'info');
            resultsDiv.style.display = 'none';

            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || 'Analysis failed');
                }

                showStatus('✓ Analysis complete!', 'success');
                displayResults(data);
                
            } catch (error) {
                showStatus('✗ Error: ' + error.message, 'error');
                console.error('Error:', error);
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Analyze Mesh';
            }
        }

        function displayResults(data) {
            let html = '<h2>Segmentation Results</h2>';
            html += `<p><strong>Total Points:</strong> ${data.num_points.toLocaleString()}</p>`;
            html += `<p><strong>Parts Detected:</strong> ${data.num_parts}</p>`;
            
            if (data.visualization_url) {
                html += `<a href="${data.visualization_url}" class="download-link" download>📥 Download Colored Point Cloud (PLY)</a>`;
            }
            
            html += '<h3>Part Breakdown</h3>';
            
            data.parts.forEach(part => {
                html += `
                    <div class="part-item">
                        <h3>Part ${part.id}: ${part.name}</h3>
                        <div class="part-stats">
                            <div class="stat">
                                <div class="stat-label">Points</div>
                                <div class="stat-value">${part.point_count.toLocaleString()}</div>
                            </div>
                            <div class="stat">
                                <div class="stat-label">Percentage</div>
                                <div class="stat-value">${part.percentage.toFixed(1)}%</div>
                            </div>
                        </div>
                        ${part.bbox ? `
                            <div style="margin-top: 10px;">
                                <strong>Bounding Box:</strong><br>
                                Size: ${part.bbox.extent[0].toFixed(3)} × ${part.bbox.extent[1].toFixed(3)} × ${part.bbox.extent[2].toFixed(3)}<br>
                                Center: (${part.bbox.center[0].toFixed(3)}, ${part.bbox.center[1].toFixed(3)}, ${part.bbox.center[2].toFixed(3)})
                            </div>
                        ` : ''}
                    </div>
                `;
            });
            
            resultsDiv.innerHTML = html;
            resultsDiv.style.display = 'block';
        }
    </script>
</body>
</html>
"""


def load_model():
    """Load segmentation backend (lazy loading)."""
    global _model, _device
    
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        backend_kind = os.environ.get("SEGMENTATION_BACKEND", "pointnet").lower()
        print(f"Loading {backend_kind} segmentation backend on {_device}...")
        
        _model = create_segmentation_backend(kind=backend_kind, device=_device)
        print("✓ Backend loaded successfully")
    
    return _model, _device


@app.route('/')
def index():
    """Main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded mesh file."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load model (using segmentation backend abstraction)
        backend_kind = os.environ.get("SEGMENTATION_BACKEND", "pointnet").lower()
        model = create_segmentation_backend(kind=backend_kind, device=device)
        
        # Run segmentation
        print(f"Analyzing mesh: {filename}")
        seg_result = model.segment(filepath, num_points=2048)
        
        points = seg_result.points
        labels = seg_result.labels
        unique_labels = np.unique(labels)
        
        # Build part statistics
        part_stats = compute_part_statistics(points, labels)
        part_bboxes = compute_part_bounding_boxes(points, labels)
        
        # Build part label names
        part_label_names = {}
        for label_id in unique_labels:
            label_id_int = int(label_id)
            result = get_category_from_flat_label(label_id_int)
            if result:
                cat, part_name = result
                part_label_names[label_id_int] = part_name
            else:
                part_label_names[label_id_int] = f"part_{label_id_int}"
        
        # Create colored point cloud
        colors = np.zeros((len(points), 3))
        for i, label_id in enumerate(labels):
            np.random.seed(int(label_id))
            color = np.random.rand(3)
            colors[i] = color
        
        pc = trimesh.PointCloud(vertices=points, colors=colors)
        viz_filename = f"segmentation_{filename.rsplit('.', 1)[0]}.ply"
        viz_path = os.path.join(app.config['UPLOAD_FOLDER'], viz_filename)
        pc.export(viz_path)
        
        # Build response
        parts = []
        for label_id in unique_labels:
            label_id_int = int(label_id)
            part_name = part_label_names.get(label_id_int, f"part_{label_id_int}")
            count = np.sum(labels == label_id_int)
            bbox = part_bboxes.get(label_id_int, {})
            
            part_data = {
                "id": int(label_id_int),
                "name": part_name,
                "point_count": int(count),
                "percentage": float(count / len(points) * 100),
            }
            
            if bbox:
                part_data["bbox"] = {
                    "min": bbox.get("min", [0, 0, 0]).tolist() if hasattr(bbox.get("min", [0, 0, 0]), "tolist") else list(bbox.get("min", [0, 0, 0])),
                    "max": bbox.get("max", [0, 0, 0]).tolist() if hasattr(bbox.get("max", [0, 0, 0]), "tolist") else list(bbox.get("max", [0, 0, 0])),
                    "center": bbox.get("center", [0, 0, 0]).tolist() if hasattr(bbox.get("center", [0, 0, 0]), "tolist") else list(bbox.get("center", [0, 0, 0])),
                    "extent": bbox.get("extent", [0, 0, 0]).tolist() if hasattr(bbox.get("extent", [0, 0, 0]), "tolist") else list(bbox.get("extent", [0, 0, 0])),
                }
            
            parts.append(part_data)
        
        response = {
            "num_points": len(points),
            "num_parts": len(unique_labels),
            "parts": parts,
            "visualization_url": f"/download/{viz_filename}",
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500


@app.route('/download/<filename>')
def download(filename):
    """Download visualization file."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    print("=" * 60)
    print("PointNet++ Segmentation Web Interface")
    print("=" * 60)
    print("Starting server on http://localhost:5001")
    print("Open this URL in your browser to test PointNet++ segmentation")
    print("=" * 60)
    print()
    
    # Pre-load model (optional, can be lazy loaded)
    try:
        load_model()
        print("✓ Model pre-loaded and ready")
    except Exception as e:
        print(f"⚠ Model will be loaded on first request: {e}")
    
    app.run(host='0.0.0.0', port=5001, debug=False)

