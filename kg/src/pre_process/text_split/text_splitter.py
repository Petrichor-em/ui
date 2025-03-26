from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


def get_text_splitter(chunk_size=5000, chunk_overlap=200, separators=["\n", "\n\n", "。"]):
    '''
    Example:
		text_splitter = get_text_splitter()
		splitted_chunks = text_splitter.split_text(text)
    '''
    # TODO: redesign this shit
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators
    )
    return text_splitter
