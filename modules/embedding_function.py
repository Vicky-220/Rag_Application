from langchain_community.embeddings import OllamaEmbeddings

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="bge-m3:latest")
    return embeddings