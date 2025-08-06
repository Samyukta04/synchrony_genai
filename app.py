from flask import Flask, render_template, request
import os
import json
from processor.main import process_uploaded_files

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    persona = request.form.get("persona", "")
    job = request.form.get("job", "")
    uploaded_files = request.files.getlist("pdfs")

    pdf_paths = []
    for file in uploaded_files:
        if file.filename.endswith(".pdf"):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            pdf_paths.append(file_path)

    output = process_uploaded_files(pdf_paths, persona, job)
    return render_template("result.html", result=output)

if __name__ == "__main__":
    app.run(debug=True)
