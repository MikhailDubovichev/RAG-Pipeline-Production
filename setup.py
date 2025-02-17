from setuptools import setup, find_packages

setup(
    name="rag-pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.100.0,<0.101.0",
        "uvicorn==0.22.0",
        "python-dotenv==1.0.0",
        "gradio>=3.30.0,<4.0.0",
        "httpx==0.28.1",
        "python-docx==1.1.2",
        "python-pptx==1.0.2",
        "openpyxl==3.1.5",
        "pandas==2.2.2",
        "pdfplumber==0.11.5",
        "llama-index==0.12.9",
        "llama-index-embeddings-nebius==0.3.1",
        "llama-index-llms-nebius==0.1.1",
        "llama-index-vector-stores-faiss==0.3.0",
        "sentence-transformers==3.3.1",
        "faiss-cpu==1.9.0.post1",
        "whoosh==2.7.4",
    ],
) 