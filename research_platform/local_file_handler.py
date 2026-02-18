import os
import pypdf

class LocalFileHandler:
    """
    Edison v4.0 Local File Ingestion Engine.
    Handles parsing and context extraction for user-uploaded files (PDF, Code, Text).
    """

    def __init__(self, upload_dir: str = "research_platform/uploads"):
        self.upload_dir = upload_dir
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

    def process_files(self, file_paths: list[str]) -> str:
        """
        Reads a list of file paths and returns a consolidated context string.
        """
        context = "### 📂 ATTACHED LOCAL FILES\n\n"

        for path in file_paths:
            if not os.path.exists(path):
                continue

            filename = os.path.basename(path)
            ext = os.path.splitext(filename)[1].lower()
            content = ""

            try:
                if ext == ".pdf":
                    content = self._read_pdf(path)
                elif ext in [".py", ".js", ".ts", ".tsx", ".md", ".json", ".txt", ".csv"]:
                    content = self._read_text(path)
                else:
                    content = f"[Binary or Unsupported Format: {ext}]"
            except Exception as e:
                content = f"[Error reading file: {str(e)}]"

            # Smart Truncation (Max 5000 chars per file to prevent overflow)
            if len(content) > 5000:
                content = content[:5000] + "\n... [Content Truncated due to size] ..."

            context += f"#### 📄 File: {filename}\n```\n{content}\n```\n\n"

        return context

    def _read_text(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _read_pdf(self, path: str) -> str:
        text = ""
        try:
            reader = pypdf.PdfReader(path)
            for page in reader.pages[:10]: # Limit to first 10 pages for speed/context
                text += page.extract_text() + "\n"
        except Exception as e:
            text = f"PDF Parse Error: {e}"
        return text
