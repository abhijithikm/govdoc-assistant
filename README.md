# GovDoc

Browser chat for questions about government PDF documents. FastAPI serves the page. The app reads each PDF locally, including scanned pages through Tesseract (Malayalam and English), and answers from the matching passages with filename and page citations.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo apt-get install tesseract-ocr tesseract-ocr-mal tesseract-ocr-eng poppler-utils
cp .env.example .env   # set NVIDIA_API_KEY
mkdir -p data          # put PDFs here; this folder is not in Git
uvicorn app:app --reload
```

Open http://127.0.0.1:8000. You can also upload PDFs from the page.

See `documentation.txt` for server deployment, indexing, and limits.
