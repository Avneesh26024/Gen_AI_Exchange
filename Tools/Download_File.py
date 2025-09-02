import os
import requests
from urllib.parse import urlparse
from langchain_core.tools import tool

# Directory where all files will be stored
DOWNLOAD_DIR = r"C:\Users\Avneesh\Desktop\HF_AI_AGENTS\Final_Assignment_Template\HF_AGENT_FILE_ANALYSIS"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

@tool
def download_file(url: str) -> str:
    """
    Download a file from the given URL and save it locally in the agent's file analysis folder.

    This tool can handle direct file links (e.g., PDF, DOCX, XLSX, CSV, TXT, audio)
    and also certain known landing-page patterns (like arXiv abstract pages).

    Args:
        url (str): URL pointing to the file or landing page.

    Returns:
        str: Local file path to the downloaded file.
             Returns an empty string if download fails.
    """
    try:
        # Handle arXiv abstract pages by converting to direct PDF
        if "arxiv.org/abs/" in url:
            url = url.replace("/abs/", "/pdf/") + ".pdf"

        # Extract file name from URL
        parsed = urlparse(url)
        file_name = os.path.basename(parsed.path)

        # If no file name found, assign default
        if not file_name:
            file_name = "downloaded_file"

        local_path = os.path.join(DOWNLOAD_DIR, file_name)

        # Stream download
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"[INFO] File downloaded: {local_path}")
        return local_path

    except Exception as e:
        print(f"[ERROR] Failed to download from {url}: {e}")
        return ""
