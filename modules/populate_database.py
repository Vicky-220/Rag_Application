from langchain.schema import Document
from modules.embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma

def add_to_vector_db(chunks: list[Document]):
    db = Chroma(
        persist_directory="./chroma_db/",
        embedding_function=get_embedding_function(),
        collection_name="pdf_chunks"
    )

    # Calculate Chunk IDs.
    chunks_with_ids = _get_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are returned by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["chunk_id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding {len(new_chunks)} new documents to DB...")
        new_chunk_ids = [chunk.metadata["chunk_id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ Database is up to date")


def _get_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Create a unique ID for the chunk.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["chunk_id"] = chunk_id

    return chunks