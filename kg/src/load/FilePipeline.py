import os
from .FileProcesser import FileProcesser
from langchain.docstore.document import Document


class FilePipeline:
    @staticmethod
    def _add_document_ids(documents: list[Document], max_doc_id: int) -> list[Document]:
        start_id = max_doc_id + 1
        for idx, doc in enumerate(documents):
            doc.metadata["document_id"] = start_id + idx
        return documents

    @classmethod
    def _process_file(cls, file_path: str, max_doc_id: int) -> list[Document]:
        if os.path.isdir(file_path):
            raise ValueError("Cann't import dir, please import a file.")
        documents = FileProcesser.load(file_path)
        return cls._add_document_ids(documents, max_doc_id)
