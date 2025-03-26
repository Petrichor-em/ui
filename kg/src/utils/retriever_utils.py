from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.llm.llm import get_llm

def init_history_retriever(config_path, stored_db):
    """
    Initialize a history-aware retriever.

    Args:
        stored_db: The database containing stored chat history.

    Returns:
        history_aware_retriever: A retriever that considers chat history.
    """
    retriever = stored_db.as_retriever(search_type="mmr", search_kwargs={"k": 1})

    contextualize_q_system_prompt = (
        "根据聊天记录以及用户的最新提问，"
        "将最新提问重新表述为一个无需借助聊天记录也能被理解的独立问题。"
        "不要回答该问题，只需进行重新表述。"
        "如果无法进行重新表述，则按原样返回该问题。"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        get_llm(config_path), retriever, contextualize_q_prompt
    )

    return history_aware_retriever
