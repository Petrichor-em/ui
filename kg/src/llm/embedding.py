import yaml
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

def get_embedding(config_path: str):
    """
    Get the embedding configuration based on the provided YAML config file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        object: An instance of the embedding class based on the configuration.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    embedding_name = config["embedding"]["use"]
    embedding_config = config["embedding"].get(embedding_name)
    
    if embedding_name is None:
        raise ValueError("Invalid embedding use in config")
    
    if embedding_name == "nomic-embed-text":
        return OllamaEmbeddings(
            model=embedding_config["model"],
            temperature=embedding_config.get("temperature"),
            base_url=embedding_config["api_base"],
        )
    elif "model" in embedding_config:
        return OpenAIEmbeddings(
            model=embedding_config["model"],
            openai_api_key=embedding_config.get("api_key"),
            openai_api_base=embedding_config["api_base"],
        )
    else:
        return OpenAIEmbeddings(
            openai_api_key=embedding_config.get("api_key"),
            openai_api_base=embedding_config["api_base"],
        )
