from typing import List, Dict, Any, Optional
import os
from colorama import init, Fore, Style
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from modules.embedding_function import get_embedding_function
from modules.pdf_loader import load_pdf_documents
from modules.text_splitter import split_documents

# Initialize colorama
init()

class DatabaseManager:
    def __init__(self):
        self.db = Chroma(
            persist_directory="./chroma_db/",
            embedding_function=get_embedding_function(),
            collection_name="pdf_chunks"
        )
        self.embeddings = get_embedding_function()

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all chunks from the database with their embeddings."""
        results = self.db.get(include=['embeddings', 'documents', 'metadatas'])
        chunks = []
        for i, (doc, metadata, embedding) in enumerate(zip(
            results['documents'], 
            results['metadatas'], 
            results['embeddings']
        )):
            chunks.append({
                'id': results['ids'][i],
                'content': doc,
                'metadata': metadata,
                'embedding': embedding
            })
        return chunks

    def get_structure(self) -> Dict[str, Dict[Any, List[Dict[str, Any]]]]:
        """Return a nested structure: {source: {page: [chunks]}}."""
        chunks = self.get_all_chunks()
        structure: Dict[str, Dict[Any, List[Dict[str, Any]]]] = {}
        for chunk in chunks:
            source = chunk['metadata'].get('source', 'unknown')
            page = chunk['metadata'].get('page', 0)
            structure.setdefault(source, {}).setdefault(page, []).append(chunk)
        return structure

    def explore_interactive(self) -> None:
        """Interactive exploration of the DB layer-by-layer: files -> pages -> chunks.

        The user can navigate back at any prompt by typing 'back'.
        """
        structure = self.get_structure()
        if not structure:
            print(f"{Fore.YELLOW}! No data found in the database{Style.RESET_ALL}")
            return

        sources = list(structure.keys())

        while True:
            print(f"\n{Fore.CYAN}Files in database:{Style.RESET_ALL}")
            for i, src in enumerate(sources, start=1):
                pages = structure[src]
                total_chunks = sum(len(chunks) for chunks in pages.values())
                print(f"{i}. {os.path.basename(src)} - Pages: {len(pages)}, Chunks: {total_chunks}")

            choice = input(f"\n{Fore.GREEN}Select a file number to inspect or 'back' to return: {Style.RESET_ALL}").strip()
            if choice.lower() == 'back':
                return
            try:
                idx = int(choice) - 1
                if idx < 0 or idx >= len(sources):
                    raise ValueError()
            except ValueError:
                print(f"{Fore.RED}Invalid selection. Please enter a file number or 'back'.{Style.RESET_ALL}")
                continue

            src = sources[idx]
            pages = structure[src]
            page_keys = sorted(pages.keys())

            # Page loop
            while True:
                print(f"\n{Fore.CYAN}Pages for {os.path.basename(src)}:{Style.RESET_ALL}")
                for j, p in enumerate(page_keys, start=1):
                    cnt = len(pages[p])
                    print(f"{j}. Page {p} - Chunks: {cnt}")

                p_choice = input(f"\n{Fore.GREEN}Select a page number to inspect, 'back' to choose another file, or 'files' to list files: {Style.RESET_ALL}").strip()
                if p_choice.lower() == 'back':
                    break
                if p_choice.lower() == 'files':
                    break
                try:
                    pidx = int(p_choice) - 1
                    if pidx < 0 or pidx >= len(page_keys):
                        raise ValueError()
                except ValueError:
                    print(f"{Fore.RED}Invalid selection. Please enter a page number, 'back', or 'files'.{Style.RESET_ALL}")
                    continue

                page_num = page_keys[pidx]
                chunks = pages[page_num]

                # Chunk loop
                while True:
                    print(f"\n{Fore.CYAN}Chunks on {os.path.basename(src)} page {page_num}:{Style.RESET_ALL}")
                    for k, chunk in enumerate(chunks, start=1):
                        preview = ' '.join(chunk['content'].split()[:5])
                        print(f"{k}. ID: {chunk['id']} - {preview}...")

                    c_choice = input(f"\n{Fore.GREEN}Enter chunk number to view full preview, 'back' to pages, or 'files' to return: {Style.RESET_ALL}").strip()
                    if c_choice.lower() == 'back':
                        break
                    if c_choice.lower() == 'files':
                        p_choice = 'files'
                        break
                    try:
                        cidx = int(c_choice) - 1
                        if cidx < 0 or cidx >= len(chunks):
                            raise ValueError()
                    except ValueError:
                        print(f"{Fore.RED}Invalid selection. Please enter a chunk number, 'back', or 'files'.{Style.RESET_ALL}")
                        continue

                    selected = chunks[cidx]
                    print(f"\n{Fore.YELLOW}--- Chunk Preview ---{Style.RESET_ALL}\n{selected['content']}\n{Fore.YELLOW}---------------------{Style.RESET_ALL}")

                if p_choice.lower() == 'files':
                    break

    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete specific chunks from the database."""
        self.db._collection.delete(ids=chunk_ids)
        self.db.persist()
        print(f"{Fore.GREEN}✓ Successfully deleted {len(chunk_ids)} chunks{Style.RESET_ALL}")

    def delete_page(self, source: str, page: int) -> None:
        """Delete all chunks from a specific page."""
        chunks = self.get_all_chunks()
        to_delete = [
            chunk['id'] for chunk in chunks
            if chunk['metadata']['source'] == source and chunk['metadata']['page'] == page
        ]
        if to_delete:
            self.delete_chunks(to_delete)
            print(f"{Fore.GREEN}✓ Successfully deleted page {page} from {source}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}! No chunks found for page {page} in {source}{Style.RESET_ALL}")

    def delete_file(self, source: str) -> None:
        """Delete all chunks from a specific file."""
        chunks = self.get_all_chunks()
        to_delete = [
            chunk['id'] for chunk in chunks
            if chunk['metadata']['source'] == source
        ]
        if to_delete:
            self.delete_chunks(to_delete)
            print(f"{Fore.GREEN}✓ Successfully deleted file {source}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}! No chunks found for file {source}{Style.RESET_ALL}")

    def add_file(self, file_path: str) -> None:
        """Add a new file to the database."""
        if not os.path.exists(file_path):
            print(f"{Fore.RED}✗ File not found: {file_path}{Style.RESET_ALL}")
            return

        try:
            # Create temporary directory loader
            from langchain.document_loaders.pdf import PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split documents
            chunks = split_documents(documents)
            
            # Add to database
            from modules.populate_database import add_to_vector_db
            add_to_vector_db(chunks)
            
            print(f"{Fore.GREEN}✓ Successfully added file: {file_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}✗ Error adding file: {str(e)}{Style.RESET_ALL}")

    def update_database(self) -> None:
        """Update database with new files from the knowledge base directory."""
        try:
            documents = load_pdf_documents()
            if documents:
                chunks = split_documents(documents)
                from modules.populate_database import add_to_vector_db
                add_to_vector_db(chunks)
        except Exception as e:
            print(f"{Fore.RED}✗ Error updating database: {str(e)}{Style.RESET_ALL}")
    
    def visualize_vectors(self) -> None:
        from sklearn.metrics.pairwise import cosine_similarity
    
        chunks = self.get_all_chunks()
        if not chunks:
            print("No vectors found.")
            return
    
        # ---------------- Data prep ----------------
        embeddings = np.array([c["embedding"] for c in chunks])
        sources = [c["metadata"]["source"] for c in chunks]
    
        pca = PCA(n_components=3)
        xyz = pca.fit_transform(embeddings)
    
        # ---------------- Stats ----------------
        unique_files = sorted(set(sources))
        total_chunks = len(chunks)
        avg_chunks = total_chunks / len(unique_files)
    
        file_chunk_count = {f: sources.count(f) for f in unique_files}
    
        # ---------------- Similarity graph ----------------
        sim = cosine_similarity(embeddings)
        np.fill_diagonal(sim, 0)
    
        k = 4
        edge_x, edge_y, edge_z = [], [], []
        total_connections = 0
    
        for i in range(len(sim)):
            nbrs = np.argsort(sim[i])[-k:]
            for j in nbrs:
                edge_x += [xyz[i, 0], xyz[j, 0], None]
                edge_y += [xyz[i, 1], xyz[j, 1], None]
                edge_z += [xyz[i, 2], xyz[j, 2], None]
                total_connections += 1
    
        # ---------------- Traces ----------------
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(color="rgba(0,255,255,0.25)", width=2),
            hoverinfo="none",
            name="Connections"
        )
    
        # Chunk index scale
        chunk_index = list(range(total_chunks))
    
        # File color mapping
        file_map = {f: i for i, f in enumerate(unique_files)}
        file_colors = [file_map[s] for s in sources]
    
        node_trace = go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker=dict(
                size=7,
                color=chunk_index,
                colorscale="Turbo",
                opacity=0.9,
                colorbar=dict(
                    title="Chunk Index",
                    thickness=18,
                    len=0.7
                )
            ),
            customdata=[
                (
                    c["id"],
                    os.path.basename(c["metadata"]["source"]),
                    c["metadata"]["page"],
                    c["content"]
                )
                for c in chunks
            ],
            hovertemplate=(
                "<b>Chunk</b><br>"
                "Index: %{marker.color}<br>"
                "File: %{customdata[1]}<br>"
                "Page: %{customdata[2]}<extra></extra>"
            ),
            name="Chunks"
        )
    
        fig = go.Figure(data=[edge_trace, node_trace])
    
        # ---------------- Side panels ----------------
        file_panel_text = "<b>FILES</b><br><br>"
        for f, c in file_chunk_count.items():
            file_panel_text += f"{os.path.basename(f)} : {c}<br>"
    
        system_panel = (
            "<b>SYSTEM</b><br><br>"
            f"Total Files: {len(unique_files)}<br>"
            f"Total Chunks: {total_chunks}<br>"
            f"Connections: {total_connections}<br>"
            f"Avg/ File: {avg_chunks:.1f}<br>"
            f"PCA Coverage: {sum(pca.explained_variance_ratio_) * 100:.1f}%"
        )
    
        # ---------------- Layout ----------------
        fig.update_layout(
            title=dict(
                text="Neural Vector Knowledge Graph",
                font=dict(color="#00FFF6", size=22),
                x=0.5
            ),
            paper_bgcolor="#05070D",
            plot_bgcolor="#05070D",
            margin=dict(l=0, r=0, t=50, b=0),
            showlegend=False,
            scene=dict(
                bgcolor="#05070D",
                xaxis=dict(showbackground=False, color="#00FFF6"),
                yaxis=dict(showbackground=False, color="#00FFF6"),
                zaxis=dict(showbackground=False, color="#00FFF6"),
                camera=dict(eye=dict(x=1.8, y=1.6, z=1.4))
            ),
            annotations=[
                # System panel
                dict(
                    text=system_panel,
                    x=1.02, y=0.85,
                    xref="paper", yref="paper",
                    showarrow=False,
                    align="left",
                    font=dict(color="#00FFF6", size=12),
                    bgcolor="rgba(5,15,25,0.65)",
                    bordercolor="#00FFF6",
                    borderwidth=1
                ),
                # File panel
                dict(
                    text=file_panel_text,
                    x=1.02, y=0.45,
                    xref="paper", yref="paper",
                    showarrow=False,
                    align="left",
                    font=dict(color="#E6FFFF", size=11),
                    bgcolor="rgba(10,20,30,0.7)",
                    bordercolor="#00FFF6",
                    borderwidth=1
                ),
                dict(
                    text="Hover = Inspect · Click = Read Chunk",
                    x=0.5, y=0.02,
                    showarrow=False,
                    font=dict(color="#00FFF6", size=11)
                )
            ],
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.02, y=0.95,
                    bgcolor="rgba(0,0,0,0.6)",
                    bordercolor="#00FFF6",
                    buttons=[
                        dict(
                            label="Color: Index",
                            method="restyle",
                            args=[{"marker.color": [chunk_index]}]
                        ),
                        dict(
                            label="Color: File",
                            method="restyle",
                            args=[{"marker.color": [file_colors]}]
                        ),
                        dict(
                            label="Color: Depth",
                            method="restyle",
                            args=[{"marker.color": [xyz[:, 2]]}]
                        ),
                        dict(
                            label="Toggle Links",
                            method="restyle",
                            args=[{"visible": [True, False]}, [0]]
                        )
                    ]
                )
            ]
        )
    
        # ---------------- Click → bottom content panel ----------------
        def on_click(trace, points, selector):
            if not points.point_inds:
                return
    
            idx = points.point_inds[0]
            c = chunks[idx]
    
            fig.update_layout(
                annotations=fig.layout.annotations + (
                    dict(
                        text=(
                            "<b>CHUNK CONTENT</b><br><br>"
                            f"{c['content']}"
                        ),
                        x=0.5, y=-0.15,
                        xref="paper", yref="paper",
                        showarrow=False,
                        align="left",
                        font=dict(color="#FFFFFF", size=11),
                        bgcolor="rgba(10,15,25,0.85)",
                        bordercolor="#00FFF6",
                        borderwidth=1
                    ),
                )
            )
    
        node_trace.on_click(on_click)
    
        fig.show()
    
    


def display_menu() -> None:
    """Display the main menu."""
    print(f"\n{Fore.CYAN}=== Database Management Menu ==={Style.RESET_ALL}")
    print(f"{Fore.WHITE}1. Delete specific chunks")
    print("2. Delete page")
    print("3. Delete file")
    print("4. Add new file")
    print("5. Update database from knowledge base")
    print("6. Visualize vectors")
    print("7. Explore database")
    print("8. Exit{Style.RESET_ALL}")

def get_user_input(prompt: str, valid_options: List[str]) -> str:
    """Get validated user input."""
    while True:
        choice = input(f"{Fore.GREEN}{prompt}: {Style.RESET_ALL}").strip()
        if choice.lower() == 'back':
            return 'back'
        if choice in valid_options:
            return choice
        print(f"{Fore.RED}Invalid choice. Please try again or type 'back' to return.{Style.RESET_ALL}")

def main():
    db_manager = DatabaseManager()
    
    while True:
        display_menu()
        choice = get_user_input("Enter your choice (or 'back' to return)", ['1', '2', '3', '4', '5', '6', '7', '8'])
        
        if choice == '8':
            break
            
        elif choice == '1':
            # Delete specific chunks
            chunks = db_manager.get_all_chunks()
            if not chunks:
                print(f"{Fore.YELLOW}No chunks found in the database{Style.RESET_ALL}")
                continue
            print(f"\n{Fore.CYAN}Available chunks:{Style.RESET_ALL}")
            for i, chunk in enumerate(chunks, start=1):
                preview = ' '.join(str(chunk['content']).split()[:6])
                print(f"{i}. ID: {chunk['id']} - Source: {os.path.basename(chunk['metadata']['source'])}, Page: {chunk['metadata']['page']} - {preview}...")

            prompt = f"{Fore.GREEN}Enter chunk numbers to delete (comma-separated), 'all' to delete all, or 'back': {Style.RESET_ALL}"
            sel = input(prompt).strip()
            if sel.lower() == 'back':
                continue
            if sel.lower() == 'all':
                ids_to_delete = [c['id'] for c in chunks]
            else:
                ids_to_delete = []
                tokens = [t.strip() for t in sel.split(',') if t.strip()]
                valid = True
                for t in tokens:
                    try:
                        idx = int(t) - 1
                        if idx < 0 or idx >= len(chunks):
                            raise ValueError()
                        ids_to_delete.append(chunks[idx]['id'])
                    except ValueError:
                        print(f"{Fore.RED}Invalid selection '{t}'. Aborting.{Style.RESET_ALL}")
                        valid = False
                        break
                if not valid:
                    continue

            confirm = input(f"{Fore.YELLOW}Confirm delete {len(ids_to_delete)} chunks? (y/n): {Style.RESET_ALL}").strip().lower()
            if confirm == 'y':
                db_manager.delete_chunks(ids_to_delete)
            else:
                print(f"{Fore.CYAN}Deletion cancelled.{Style.RESET_ALL}")

        elif choice == '2':
            # Delete page - show files then pages for selection
            structure = db_manager.get_structure()
            if not structure:
                print(f"{Fore.YELLOW}No data found in the database{Style.RESET_ALL}")
                continue

            sources = list(structure.keys())
            print(f"\n{Fore.CYAN}Files in database:{Style.RESET_ALL}")
            for i, src in enumerate(sources, start=1):
                pages = structure[src]
                total_chunks = sum(len(chunks) for chunks in pages.values())
                print(f"{i}. {os.path.basename(src)} - Pages: {len(pages)}, Chunks: {total_chunks}")

            s_choice = input(f"{Fore.GREEN}Select file number to delete a page from or 'back': {Style.RESET_ALL}").strip()
            if s_choice.lower() == 'back':
                continue
            try:
                sidx = int(s_choice) - 1
                src = sources[sidx]
            except Exception:
                print(f"{Fore.RED}Invalid selection{Style.RESET_ALL}")
                continue

            page_keys = sorted(structure[src].keys())
            print(f"\n{Fore.CYAN}Pages for {os.path.basename(src)}:{Style.RESET_ALL}")
            for j, p in enumerate(page_keys, start=1):
                print(f"{j}. Page {p} - Chunks: {len(structure[src][p])}")

            p_choice = input(f"{Fore.GREEN}Select page number to delete or 'back': {Style.RESET_ALL}").strip()
            if p_choice.lower() == 'back':
                continue
            try:
                pidx = int(p_choice) - 1
                page_num = page_keys[pidx]
            except Exception:
                print(f"{Fore.RED}Invalid page selection{Style.RESET_ALL}")
                continue

            confirm = input(f"{Fore.YELLOW}Confirm delete page {page_num} from {os.path.basename(src)}? (y/n): {Style.RESET_ALL}").strip().lower()
            if confirm == 'y':
                db_manager.delete_page(src, int(page_num))
            else:
                print(f"{Fore.CYAN}Deletion cancelled.{Style.RESET_ALL}")

        elif choice == '3':
            # Delete file - show files for selection
            structure = db_manager.get_structure()
            if not structure:
                print(f"{Fore.YELLOW}No data found in the database{Style.RESET_ALL}")
                continue

            sources = list(structure.keys())
            print(f"\n{Fore.CYAN}Files in database:{Style.RESET_ALL}")
            for i, src in enumerate(sources, start=1):
                pages = structure[src]
                total_chunks = sum(len(chunks) for chunks in pages.values())
                print(f"{i}. {os.path.basename(src)} - Pages: {len(pages)}, Chunks: {total_chunks}")

            s_choice = input(f"{Fore.GREEN}Select file number to delete or 'back': {Style.RESET_ALL}").strip()
            if s_choice.lower() == 'back':
                continue
            try:
                sidx = int(s_choice) - 1
                src = sources[sidx]
            except Exception:
                print(f"{Fore.RED}Invalid selection{Style.RESET_ALL}")
                continue

            confirm = input(f"{Fore.YELLOW}Confirm delete file {os.path.basename(src)} and all its chunks? (y/n): {Style.RESET_ALL}").strip().lower()
            if confirm == 'y':
                db_manager.delete_file(src)
            else:
                print(f"{Fore.CYAN}Deletion cancelled.{Style.RESET_ALL}")

        elif choice == '4':
            # Add new file
            file_path = input(f"{Fore.GREEN}Enter the full path to the PDF file or 'back': {Style.RESET_ALL}").strip()
            if file_path.lower() == 'back':
                continue
                
            db_manager.add_file(file_path)

        elif choice == '5':
            # Update database
            confirm = input(f"{Fore.YELLOW}This will scan the knowledge base directory for new files. Continue? (y/n): {Style.RESET_ALL}").strip().lower()
            if confirm == 'y':
                db_manager.update_database()

        elif choice == '6':
            # Visualize vectors
            print(f"{Fore.CYAN}Opening vector visualization in browser...{Style.RESET_ALL}")
            db_manager.visualize_vectors()
        elif choice == '7':
            # Explore database
            db_manager.explore_interactive()

if __name__ == "__main__":
    main()
