"""
Interactive Database Management CLI & Vector Visualizer
Allows inspecting, exploring, querying, deleting, refreshing, and visualizing the Chroma vector database.
"""
import os
import sys
import webbrowser
from pathlib import Path
from colorama import init, Fore, Style

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.modules.vector_db import VectorStore
from backend.modules.pdf_loader import load_pdf_documents
from backend.modules.text_splitter import split_documents
from backend.modules.visualizer import VectorVisualizer
from backend.config.settings import KNOWLEDGE_BASE_DIR

init()


def print_banner():
    print(f"{Fore.CYAN}{Style.BRIGHT}")
    print("=" * 60)
    print("      LOCAL MULTI-AGENTIC RAG - DATABASE MANAGER      ")
    print("=" * 60)
    print(f"{Style.RESET_ALL}")


def display_menu():
    print(f"\n{Fore.YELLOW}Database Management Menu:{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}1.{Style.RESET_ALL} View Collection Statistics")
    print(f"  {Fore.GREEN}2.{Style.RESET_ALL} Interactive Exploration (Files -> Pages -> Chunks)")
    print(f"  {Fore.GREEN}3.{Style.RESET_ALL} Search Vector Database")
    print(f"  {Fore.GREEN}4.{Style.RESET_ALL} Populate / Refresh Database from knowledge_base/")
    print(f"  {Fore.GREEN}5.{Style.RESET_ALL} Delete a File from Database")
    print(f"  {Fore.GREEN}6.{Style.RESET_ALL} Delete a Specific Page from Database")
    print(f"  {Fore.GREEN}7.{Style.RESET_ALL} Reset / Clear Vector Database")
    print(f"  {Fore.GREEN}8.{Style.RESET_ALL} Generate & Open 3D Vector Space Visualization")
    print(f"  {Fore.RED}0.{Style.RESET_ALL} Exit")


def view_stats(vs: VectorStore):
    stats = vs.get_collection_stats()
    print(f"\n{Fore.CYAN}--- Collection Statistics ---{Style.RESET_ALL}")
    print(f"Total Chunks:  {Fore.GREEN}{stats['total_chunks']}{Style.RESET_ALL}")
    print(f"Total Sources: {Fore.GREEN}{stats['total_sources']}{Style.RESET_ALL}")
    print(f"Sources:")
    for src in stats["sources"]:
        print(f"  - {os.path.basename(src)} ({src})")


def explore_interactive(vs: VectorStore):
    structure = vs.get_structure()
    if not structure:
        print(f"{Fore.YELLOW}! No data found in the database{Style.RESET_ALL}")
        return

    sources = list(structure.keys())

    while True:
        print(f"\n{Fore.CYAN}Files in database:{Style.RESET_ALL}")
        for i, src in enumerate(sources, start=1):
            pages = structure[src]
            total_chunks = sum(len(chunks) for chunks in pages.values())
            print(f"  {i}. {os.path.basename(src)} - Pages: {len(pages)}, Chunks: {total_chunks}")

        choice = input(f"\n{Fore.GREEN}Select a file number (or 'back'): {Style.RESET_ALL}").strip()
        if choice.lower() in ('back', 'exit', 'q'):
            return
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(sources):
                raise ValueError()
        except ValueError:
            print(f"{Fore.RED}Invalid selection.{Style.RESET_ALL}")
            continue

        src = sources[idx]
        pages = structure[src]
        page_keys = sorted(pages.keys())

        while True:
            print(f"\n{Fore.CYAN}Pages for {os.path.basename(src)}:{Style.RESET_ALL}")
            for j, p in enumerate(page_keys, start=1):
                cnt = len(pages[p])
                print(f"  {j}. Page {p} - Chunks: {cnt}")

            p_choice = input(f"\n{Fore.GREEN}Select a page number (or 'back'): {Style.RESET_ALL}").strip()
            if p_choice.lower() in ('back', 'exit', 'q'):
                break
            try:
                p_idx = int(p_choice) - 1
                if p_idx < 0 or p_idx >= len(page_keys):
                    raise ValueError()
            except ValueError:
                print(f"{Fore.RED}Invalid page selection.{Style.RESET_ALL}")
                continue

            page_num = page_keys[p_idx]
            chunks = pages[page_num]

            print(f"\n{Fore.CYAN}Chunks on Page {page_num} ({len(chunks)} chunks):{Style.RESET_ALL}")
            for k, chunk in enumerate(chunks, start=1):
                cid = chunk["metadata"].get("chunk_id", chunk["id"])
                print(f"\n{Fore.YELLOW}Chunk {k} [{cid}]:{Style.RESET_ALL}")
                print(chunk["content"].strip()[:400] + "...")


def search_db(vs: VectorStore):
    query = input(f"\n{Fore.GREEN}Enter search query: {Style.RESET_ALL}").strip()
    if not query:
        return
    results = vs.search(query, k=5, score_threshold=0.1)
    if not results:
        print(f"{Fore.YELLOW}No matching chunks found.{Style.RESET_ALL}")
        return

    print(f"\n{Fore.CYAN}Found {len(results)} matches:{Style.RESET_ALL}")
    for i, res in enumerate(results, 1):
        print(f"\n{Fore.MAGENTA}[{i}] Score: {res['score']} | File: {res['source']} (Page {res['page']}){Style.RESET_ALL}")
        print(f"Chunk ID: {res['chunk_id']}")
        print(res["content"][:300] + "...")


def refresh_db(vs: VectorStore):
    print(f"\n{Fore.CYAN}Loading documents from {KNOWLEDGE_BASE_DIR}...{Style.RESET_ALL}")
    docs = load_pdf_documents()
    if not docs:
        print(f"{Fore.RED}No PDF documents found in {KNOWLEDGE_BASE_DIR}.{Style.RESET_ALL}")
        return
    print(f"Loaded {len(docs)} document pages. Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"Generated {len(chunks)} chunks. Adding to Chroma...")
    added = vs.add_documents(chunks)
    print(f"{Fore.GREEN}✅ Done! Added {added} new chunks.{Style.RESET_ALL}")


def delete_file_cli(vs: VectorStore):
    stats = vs.get_collection_stats()
    sources = stats["sources"]
    if not sources:
        print(f"{Fore.YELLOW}Database is empty.{Style.RESET_ALL}")
        return
    print(f"\n{Fore.CYAN}Select file to delete:{Style.RESET_ALL}")
    for i, src in enumerate(sources, 1):
        print(f"  {i}. {os.path.basename(src)}")
    choice = input(f"{Fore.GREEN}Enter number (or 'back'): {Style.RESET_ALL}").strip()
    if choice.lower() in ('back', 'q'):
        return
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(sources):
            target = sources[idx]
            deleted = vs.delete_file(target)
            print(f"{Fore.GREEN}Deleted {deleted} chunks for file {target}.{Style.RESET_ALL}")
    except ValueError:
        print(f"{Fore.RED}Invalid input.{Style.RESET_ALL}")


def delete_page_cli(vs: VectorStore):
    source = input(f"{Fore.GREEN}Enter filename (or full path): {Style.RESET_ALL}").strip()
    if not source:
        return
    try:
        page = int(input(f"{Fore.GREEN}Enter page number: {Style.RESET_ALL}").strip())
        deleted = vs.delete_page(source, page)
        print(f"{Fore.GREEN}Deleted {deleted} chunks for {source} on page {page}.{Style.RESET_ALL}")
    except ValueError:
        print(f"{Fore.RED}Invalid page number.{Style.RESET_ALL}")


def reset_db_cli(vs: VectorStore):
    confirm = input(f"{Fore.RED}WARNING: This will delete all vectors in the collection. Confirm? (yes/no): {Style.RESET_ALL}").strip()
    if confirm.lower() == 'yes':
        if vs.reset_database():
            print(f"{Fore.GREEN}Database successfully cleared!{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Failed to reset database.{Style.RESET_ALL}")


def visualize_cli(vs: VectorStore):
    print(f"\n{Fore.CYAN}Generating 3D vector space visualization...{Style.RESET_ALL}")
    viz = VectorVisualizer(vs)
    html_content = viz.generate_html_plot()

    out_file = PROJECT_ROOT / "vector_space_3d.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"{Fore.GREEN}Saved visualization to: {out_file}{Style.RESET_ALL}")
    try:
        webbrowser.open(f"file://{out_file.absolute()}")
        print(f"{Fore.GREEN}Opened in browser!{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.YELLOW}Could not automatically open browser: {e}. You can open {out_file} directly.{Style.RESET_ALL}")


def main():
    print_banner()
    vs = VectorStore()

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ('--visualize', '-v'):
            visualize_cli(vs)
            return
        elif arg in ('--refresh', '-r'):
            refresh_db(vs)
            return
        elif arg in ('--stats', '-s'):
            view_stats(vs)
            return

    while True:
        display_menu()
        choice = input(f"\n{Fore.GREEN}Enter option [0-8]: {Style.RESET_ALL}").strip()

        if choice == '1':
            view_stats(vs)
        elif choice == '2':
            explore_interactive(vs)
        elif choice == '3':
            search_db(vs)
        elif choice == '4':
            refresh_db(vs)
        elif choice == '5':
            delete_file_cli(vs)
        elif choice == '6':
            delete_page_cli(vs)
        elif choice == '7':
            reset_db_cli(vs)
        elif choice == '8':
            visualize_cli(vs)
        elif choice in ('0', 'q', 'exit'):
            print(f"\n{Fore.CYAN}Exiting Database Manager. Goodbye!{Style.RESET_ALL}")
            break
        else:
            print(f"{Fore.RED}Invalid option. Please choose between 0 and 8.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
