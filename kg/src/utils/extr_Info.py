from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from tabulate import tabulate
from langchain.chains.summarize import load_summarize_chain
import textwrap
from langchain.schema import Document
import sys, os

# Add the project root directory to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(project_root, "src")):
    project_root = os.path.dirname(project_root)
sys.path.append(project_root)

from src.llm.llm import get_llm
from src.pre_process.text_split import text_splitter


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_sum(config_path:str, full_context: str):
    """
    Summarize the full text.

    Args:
        full_context (str): Full text content.

    Returns:
        str: Text summary.
    """
    contents = text_splitter.split_text(full_context)

    initial_prompt_template = """Please summarize the following content in Chinese:
    {content}
    Summary:"""
    initial_prompt = PromptTemplate(
        template=initial_prompt_template, input_variables=["content"]
    )

    refine_template = """
    Existing summary:
    {existing_answer}
    New content:
    ------------
    {content}
    ------------
    Improve the existing summary based on the new content:
    """
    refine_prompt = PromptTemplate(
        template=refine_template, input_variables=["existing_answer", "content"]
    )

    chain = load_summarize_chain(
        get_llm(config_path),
        chain_type="refine",
        question_prompt=initial_prompt,
        refine_prompt=refine_prompt,
        document_variable_name="content",
    )

    documents = [Document(page_content=text) for text in contents]

    output_summary = chain.run(documents)
    report_sum = textwrap.fill(output_summary, width=100)

    return report_sum


def extract_metadata(config_path:str, content: str):
    """
    Extract key information from the report, including document name, document ID, and document summary.

    Args:
        content (str): Text content to extract metadata from.

    Returns:
        dict: Dictionary containing document metadata.
    """
    keyinfos = "Document name, Document ID, Document summary"

    template = """
    Please use the following context to answer the questions at the end. Answer each question in sequence, and return the results in the specified format. If you don't know the answer, say you don't know. Answer in Chinese and be as detailed as possible.
    Context: {context}
    Question: Please tell me {question}
    Format: {output}
    Note: 1. The document name usually appears at the beginning of the document. If you cannot extract a suitable document name, create a Chinese document name based on the content.
    """

    class MetadataInfo(BaseModel):
        document_name: str = Field(default="", description="Document name")
        document_id: str = Field(default="", description="Document ID")
        document_summary: str = Field(default="", description="Document summary")

    output_parser = PydanticOutputParser(pydantic_object=MetadataInfo)
    prompt = PromptTemplate.from_template(template)
    parser_instructions = output_parser.get_format_instructions()

    chunks = text_splitter.split_text(content)
    metadata = MetadataInfo()

    for chunk in chunks:
        final_prompt = prompt.invoke(
            {"context": chunk, "output": parser_instructions, "question": keyinfos}
        )
        answer_chain = get_llm(config_path=config_path) | StrOutputParser()

        response = answer_chain.invoke(final_prompt)
        current_metadata = output_parser.invoke(response)

        for field in metadata.__fields__.keys():
            current_value = getattr(current_metadata, field)
            current_value = current_value.replace(" ", "").strip()

            if current_value and not getattr(metadata, field).strip():
                setattr(metadata, field, current_value)

    metadata_dict = metadata.dict()

    return metadata_dict


def pretty_print_table(input_content):
    print("\n")
    print(tabulate(input_content, headers=["Field", "Content"], tablefmt="grid"))
