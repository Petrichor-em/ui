import os
import subprocess
import tempfile
import shutil
from typing import Optional
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document

class FileProcesser(BaseLoader):
    @staticmethod
    def _get_file_extension(file_path: str) -> str:
        return os.path.splitext(file_path)[-1].lower()

    @staticmethod
    def _load_markdown(file_path: str) -> list[Document]:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        metadata = {"source": file_path}
        return [Document(page_content=content, metadata=metadata)]

    @staticmethod
    def _load_pdf(file_path: str) -> list[Document]:
        from langchain_community.document_loaders import PDFMinerLoader
        return PDFMinerLoader(file_path).load()
    
    @staticmethod
    def _load_word(file_path: str) -> list[Document]:
        '''
        Turn docx into html, then turn html into markdown.
        '''
        import mammoth
        from markdownify import markdownify as md
        with open(file_path, "rb") as docx_file:
            result = mammoth.convert_to_html(docx_file)
            html_content = result.value

        md_content = md(html_content)
        return [Document(page_content=md_content, metadata={"source": file_path})]

    @staticmethod
    def _convert_pdf_to_md(file_path: str) -> Optional[str]:
        """
        Convert a PDF file to a Markdown file with images.

        Args:
            pdf_file_name (str): Path to the input PDF file.

        Returns:
            str: Path to the generated Markdown file.
        """
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod
        try:
            # set output mkdir
            file_name = os.path.basename(file_path.split(".")[0])
            base_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../../output"
            )
            local_md_dir = os.path.join(base_path, file_name, "file")
            local_image_dir = os.path.join(base_path, file_name, "images")
            name_without_suff = os.path.basename(file_path.split(".")[0])
            image_dir = str(os.path.basename(local_image_dir))
            name_without_suff = os.path.join(local_md_dir, name_without_suff)

            os.makedirs(local_md_dir, exist_ok=True)
            os.makedirs(local_image_dir, exist_ok=True)

            image_writer, md_writer = FileBasedDataWriter(
                local_image_dir
            ), FileBasedDataWriter(local_md_dir)

            # read pdf bytes
            reader1 = FileBasedDataReader("")
            pdf_bytes = reader1.read(file_path)
            ds = PymuDocDataset(pdf_bytes)

            convert_path = f"{name_without_suff}.md"
            # inference
            if ds.classify() == SupportedPdfParseMethod.OCR:
                ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer).dump_md(
                    md_writer, convert_path, image_dir
                )
            else:
                ds.apply(doc_analyze, ocr=False).pipe_txt_mode(image_writer).dump_md(
                    md_writer, convert_path, image_dir
                )
            print(local_md_dir)
            print(local_image_dir)
            print(convert_path)
            return convert_path
        except Exception as e:
            print(f"Failure to convert PDF: {str(e)}")
            return None

    @staticmethod
    def _convert_word_to_pdf(file_path: str) -> Optional[str]:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                command = [
                    "libreoffice",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    file_path,
                    "--outdir",
                    temp_dir,
                ]
                subprocess.run(command, check=True, capture_output=True)

                # Get generated PDF path
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                pdf_filename = f"{base_name}.pdf"
                pdf_path = os.path.join(temp_dir, pdf_filename)

                if not os.path.exists(pdf_path):
                    raise ValueError(f"Conversion failed: {pdf_path} not created")

                # Create named temporary file to return
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as tmp_file:
                    # Copy PDF content to new temp file
                    with open(pdf_path, "rb") as src_file:
                        shutil.copyfileobj(src_file, tmp_file)
                    converted_path = tmp_file.name

                return converted_path
        except Exception as e:
            print(f"File conversion failed: {str(e)}")
            return None

    @classmethod
    def load(cls, file_path: str, **kwargs) -> list[Document]:
        '''
        Load a file and return Document.
        Args:
            file_path (str): The path to the file to load.
            **kwargs: Additional arguments to pass to the loader.
        Returns:
            list[Document]: A list of documents.
        '''
        
        ext = cls._get_file_extension(file_path)

        if ext == ".md":
            return cls._load_markdown(file_path)
        elif ext == ".pdf":
            md_path = cls._convert_pdf_to_md(file_path)
            return cls._load_markdown(md_path)
        elif ext in (".docx", ".doc"):
            if kwargs.get("convert_to_pdf", False):
                pdf_path = cls._convert_word_to_pdf(file_path)
                md_path = cls._convert_pdf_to_md(pdf_path)
                return cls._load_markdown(md_path)
            else:
                return cls._load_word(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
