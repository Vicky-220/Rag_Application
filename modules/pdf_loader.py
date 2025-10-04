from langchain.document_loaders.pdf import PyPDFDirectoryLoader

PATH = "./knowledge_base/"
def load_pdf_documents():
    try:
        document_loader = PyPDFDirectoryLoader(PATH)
        return document_loader.load()
    except Exception as e:
        print(f"Error loading PDF documents: {e}")
        return None

# if __name__ == '__main__':
#     documents = load_pdf_documents()
#     if documents:
#         print(documents[3])