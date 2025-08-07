import gradio as gr

def qa_interface(file, query):
    path = f"data/{file.name}"
    file.save(path)
    return handle_file(path, query)

gr.Interface(fn=qa_interface, 
             inputs=["file", "text"], 
             outputs="text",
             title="Financial Document Q&A Assistant").launch()

from processor.pdf_handler import extract_pdf_chunks
from processor.csv_handler import extract_csv_chunks
from processor.embedder import get_top_chunks
from models.summarizer import summarize_text  # optional
import os

def handle_file(file_path, query):
    ext = os.path.splitext(file_path)[1]
    if ext == ".pdf":
        chunks = extract_pdf_chunks(file_path)
    elif ext == ".csv":
        chunks = extract_csv_chunks(file_path)
    else:
        return "Unsupported file type"

    top_chunks = get_top_chunks(chunks, query, k=3)

    response = "\n\n".join([f"{c['text']} (Source: {c['source']})" for c in top_chunks])
    return response  # or summarize_text(response)

# Fast testing
if __name__ == "__main__":
    path = "data/sample_report.pdf"
    question = "What was the revenue in March?"
    print(handle_file(path, question))
