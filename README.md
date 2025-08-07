# PDF Information Extractor Web Application

A powerful web application that extracts specific information from PDF documents using advanced text processing and similarity matching techniques.

## Features

- **Upload Multiple PDFs**: Support for multiple PDF files with drag-and-drop interface
- **Smart Query Processing**: Natural language queries to extract specific information
- **Intelligent Text Extraction**: Advanced PDF parsing with section detection
- **Similarity Matching**: Uses embeddings (Ollama) or fallback similarity algorithms
- **Real-time Processing**: Live progress updates and responsive UI
- **Detailed Results**: Comprehensive answers with source attribution and relevance scores

## Prerequisites

1. **Python 3.8+**
2. **Ollama** (optional, for better embeddings)
   - Install from: https://ollama.ai
   - Recommended models: `nomic-embed-text`, `all-minilm`, `mxbai-embed-large`

## Installation & Setup

### 1. Clone/Create the Project Structure

```bash
mkdir SYNCHRONY_GENAI
cd SYNCHRONY_GENAI
```

### 2. Create Virtual Environment

```bash
python -m venv pdf_env
source pdf_env/bin/activate  # On Windows: pdf_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Set Up Ollama (Optional but Recommended)

```bash
# Install Ollama from https://ollama.ai
# Then pull a good embedding model:
ollama pull nomic-embed-text
# or
ollama pull all-minilm
```

### 6. Create Directory Structure

```bash
mkdir -p uploads processed static/uploads templates static/css static/js
```

### 7. Run the Application

```bash
python app.py
```

The application will be available at: **http://localhost:5000**

## Project Structure

```
pdf_processor_webapp/
├── app.py                 # Flask web application
├── pdf_processor.py       # Core PDF processing logic
├── requirements.txt       # Python dependencies
├── uploads/              # Temporary uploaded files
├── processed/            # Processed results storage
├── static/
│   ├── css/
│   │   └── style.css    # Custom styling
│   ├── js/
│   │   └── main.js      # Frontend JavaScript
│   └── uploads/         # Static uploads directory
├── templates/
│   ├── base.html        # Base HTML template
│   └── index.html       # Main interface
└── README.md           # This file
```

## Usage

1. **Access the Web Interface**: Open http://localhost:5000 in your browser

2. **Upload PDF Files**: 
   - Click "Select PDF Files" or drag and drop
   - Multiple files supported (max 50MB each)

3. **Enter Your Query**: 
   - Be specific about what information you want
   - Examples: "What restaurants are recommended?", "Travel tips for students"

4. **Extract Information**: 
   - Click "Extract Information"
   - Wait for processing (may take 30-60 seconds)
   - View results with source attribution

## Example Queries

- "What are the best restaurants mentioned?"
- "Travel tips for college students"
- "Hotel recommendations and accommodations"
- "Cultural attractions and historical sites"
- "Nightlife and entertainment options"
- "Packing tips and travel advice"

## Technical Details

### PDF Processing
- Uses PyMuPDF (fitz) for PDF parsing
- Intelligent heading detection
- Section-based content extraction
- Smart title generation

### Text Similarity
- Primary: Ollama embedding models
- Fallback: Custom embedding algorithm with travel-specific keywords
- Cosine similarity matching
- Document diversity scoring

### Web Interface
- Flask backend with file upload handling
- Bootstrap 5 responsive design
- Real-time processing feedback
- AJAX-based form submission
- Progressive result display

## Configuration

### Environment Variables (Optional)
```bash
export FLASK_ENV=development  # For debug mode
export MAX_FILE_SIZE=52428800  # 50MB in bytes
export UPLOAD_TIMEOUT=300     # 5 minutes
```

### Customizing Processing
Edit `pdf_processor.py` to:
- Modify keyword sets for different domains
- Adjust similarity thresholds
- Change section extraction logic
- Add new embedding models

## Troubleshooting

### Common Issues

1. **"No embedding models available"**
   - Install Ollama and pull an embedding model
   - Or rely on the fallback embedding system

2. **"spaCy model not found"**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **File upload errors**
   - Check file size (max 50MB per file)
   - Ensure files are valid PDFs
   - Check disk space in upload directory

4. **Processing timeout**
   - Large files may take longer
   - Check Ollama service status
   - Monitor server logs for specific errors

### Performance Tips

1. **Use Ollama embedding models** for better accuracy
2. **Limit file sizes** for faster processing
3. **Be specific in queries** for better results
4. **Use SSD storage** for faster file operations

## Development

### Adding New Features

1. **Backend changes**: Modify `pdf_processor.py` or `app.py`
2. **Frontend changes**: Edit templates and static files
3. **Styling**: Update `static/css/style.css`
4. **JavaScript**: Modify `static/js/main.js`

### API Endpoints

- `POST /upload`: Upload files and process query
- `GET /results/<session_id>`: Retrieve processing results
- `GET /process_status`: Check processing status

### Testing

```bash
# Test with sample PDFs
curl -X POST -F "files[]=@sample.pdf" -F "query=test query" http://localhost:5000/upload
```

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review server logs for specific error messages
3. Ensure all dependencies are properly installed
4. Verify PDF files are not corrupted

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.