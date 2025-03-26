def format_docs(docs):
    return "\n\n".join(
        f"{doc.page_content}\n\nMetadata: {doc.metadata}" for doc in docs
    )

def pretty_print(documents):
    """
    Print retrieved document content.

    Args:
        documents (list): List of documents.

    Returns:
        None
    """
    print("----------Vector Database Retrieval Results--------\n")
    for i in range(len(documents)):
        print(f"Retrieved Document - {i}\n\n")
        print(documents[i].page_content)
        print("-" * 30)
