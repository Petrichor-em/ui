from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.messages import BaseMessage

from app.qa_chatgpt.database.models import Session, get_db


def load_qa_history(session_id: str) -> BaseChatMessageHistory:
    """
    Load chat history from the database.

    Args:
        session_id (str): Session ID.

    Returns:
        BaseChatMessageHistory: Chat message history.
    """
    db = next(get_db())
    chat_history = ChatMessageHistory()
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if session:
            for message in session.messages:
                if message.role == "assistant":
                    chat_history.add_message(AIMessage(content=message.content))
                elif message.role == "user":
                    chat_history.add_message(HumanMessage(content=message.content))
                elif message.role == "system":
                    chat_history.add_message(SystemMessage(content=message.content))
                elif message.role == "tool":
                    chat_history.add_message(ToolMessage(content=message.content))
    except SQLAlchemyError:
        pass
    finally:
        db.close()
    return chat_history


def trimmer(message: BaseChatMessageHistory) -> BaseChatMessageHistory:
    """
    Trim message history to ensure token count does not exceed the limit.

    Args:
        message (BaseChatMessageHistory): Chat message history.

    Returns:
        BaseChatMessageHistory: Trimmed chat message history.
    """
    from langchain_core.messages import trim_messages
    import tiktoken

    def str_token_counter(text: str) -> int:
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))

    def tiktoken_counter(messages: List[BaseMessage]) -> int:
        num_tokens = 3
        tokens_per_message = 3
        tokens_per_name = 1
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, ToolMessage):
                role = "tool"
            elif isinstance(msg, SystemMessage):
                role = "system"
            num_tokens += (
                tokens_per_message
                + str_token_counter(role)
                + str_token_counter(msg.content)
            )
            if msg.name:
                num_tokens += tokens_per_name + str_token_counter(msg.name)
        return num_tokens

    message.messages = trim_messages(
        message.messages,
        token_counter=tiktoken_counter,
        strategy="last",
        max_tokens=5000,
        start_on="human",
        end_on=("human", "ai"),
        include_system=True,
    )
    return message
