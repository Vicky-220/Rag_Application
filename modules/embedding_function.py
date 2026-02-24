from langchain_community.embeddings import OllamaEmbeddings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="bge-m3:latest")
    return embeddings