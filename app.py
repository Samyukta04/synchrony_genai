import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename
from pdf_processor import PDFProcessor

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'pdf'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize processor
processor = PDFProcessor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files selected'}), 400
    
    files = request.files.getlist('files[]')
    query = request.form.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Please enter a query'}), 400
    
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': 'No files selected'}), 400
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_folder, exist_ok=True)
    
    uploaded_files = []
    
    # Save uploaded files
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to avoid conflicts
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            unique_filename = f"{timestamp}{filename}"
            file_path = os.path.join(session_folder, unique_filename)
            file.save(file_path)
            uploaded_files.append(file_path)
    
    if not uploaded_files:
        return jsonify({'error': 'No valid PDF files uploaded'}), 400
    
    try:
        # Process files
        results = processor.process_files(uploaded_files, query)
        
        # Save results
        results_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{session_id}.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    finally:
        # Clean up uploaded files
        try:
            import shutil
            shutil.rmtree(session_folder)
        except:
            pass

@app.route('/results/<session_id>')
def get_results(session_id):
    try:
        results_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{session_id}.json")
        if not os.path.exists(results_path):
            return jsonify({'error': 'Results not found'}), 404
        
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve results: {str(e)}'}), 500

@app.route('/process_status')
def process_status():
    """Check if processing is complete"""
    return jsonify({'status': 'ready'})

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB per file.'}), 413

@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting PDF Processor Web Application...")
    print("Available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)