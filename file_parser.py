import os
import logging
from typing import Optional
import pypdf

logger = logging.getLogger(__name__)

class FileParser:
    """Parses various file formats to extract text content."""

    SUPPORTED_TEXT_EXTENSIONS = {
        '.txt', '.md', '.py', '.js', '.json', '.html', '.xml', 
        '.css', '.csv', '.yaml', '.yml', '.sh', '.bat', '.log',
        '.ini', '.cfg', '.conf'
    }

    @staticmethod
    def get_file_extension(file_path: str) -> str:
        return os.path.splitext(file_path)[1].lower()

    @classmethod
    def parse_file(cls, file_path: str) -> Optional[str]:
        """
        Determines the file type and extracts text content.
        Returns the extracted text or None if the file type is unsupported or extraction fails.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None

        ext = cls.get_file_extension(file_path)

        try:
            if ext == '.pdf':
                return cls._parse_pdf(file_path)
            elif ext in cls.SUPPORTED_TEXT_EXTENSIONS:
                return cls._parse_text(file_path)
            else:
                # Try parsing as text if it's not a known binary format
                # This is a bit risky but useful for code files with weird extensions
                return cls._parse_text(file_path)
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return f"[Error parsing file: {e}]"

    @staticmethod
    def _parse_pdf(file_path: str) -> str:
        """Extracts text from a PDF file."""
        text_content = []
        try:
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)
            return "\n".join(text_content)
        except Exception as e:
            raise Exception(f"PDF parsing failed: {str(e)}")

    @staticmethod
    def _parse_text(file_path: str) -> str:
        """Extracts text from a text-based file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Text parsing failed: {str(e)}")
