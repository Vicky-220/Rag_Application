"""
Text splitter module for splitting documents into manageable chunks
"""
import warnings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=DeprecationWarning)


def split_documents(documents: list[Document], chunk_size: int = 600, chunk_overlap: int = 200):
    """
    Split documents into chunks using RecursiveCharacterTextSplitter
    
    Args:
        documents (list[Document]): List of documents to split
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list[Document]: List of split document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)
