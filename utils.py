import tiktoken
from urllib.parse import urljoin
from tqdm import tqdm
from openai import OpenAI
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests
from colorama import Fore, Style
import faiss
import openai
import numpy as np
import json
import os

from dotenv import load_dotenv
load_dotenv()


SHOPIFY_PASSWORD = os.getenv("SHOPIFY_PASSWORD")


def build_rag(rag_folder, url, is_shopify=False):
    # scrap websites
    if not os.path.exists(f"{rag_folder}/rag_context.json"):
        os.makedirs(rag_folder)

        scraped_text = scrape_page_for_rag(
            url, max_depth=20, is_shopify=is_shopify
        )

        # save context
        with open(f"{rag_folder}/rag_context.json", "w", encoding="utf-8") as f:
            json.dump(scraped_text, f, indent=2)

    # Create and save FAISS vector DB
    if not os.path.exists(f"{rag_folder}/faiss_index.index"):
        with open(f"{rag_folder}/rag_context.json", "r", encoding="utf-8") as f:
            blocks = json.load(f)
        create_vector_db(
            blocks, index_path=f"{rag_folder}/faiss_index.index", meta_path=f"{rag_folder}/metadata.json"
        )


def authenticate_shopify_storefront(session, url, password):
    password_url = f"{url.rstrip('/')}/password"
    payload = {"password": password}
    try:
        response = session.post(password_url, data=payload, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Authentication failed: {e}")


visited = set()


def scrape_page_for_rag(url: str, depth=0, max_depth=2, results=None, session=None, is_shopify=False):
    if results is None:
        results = []

    if session is None:
        session = requests.Session()
        if is_shopify:
            authenticate_shopify_storefront(session, url, SHOPIFY_PASSWORD)

    if url in visited or depth > max_depth:
        return results

    visited.add(url)

    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()

        text = extract_text_with_links(soup, url)

        if text:
            combined = f"[URL]: {url}\n[CONTENT]:\n{text}"
            results.append(combined)
            print(f"Scraped {url} (depth {depth}, {len(text)} characters)")

        # Follow internal links
        for link in soup.find_all('a', href=True):
            next_url = urljoin(url, link['href'])
            if urlparse(next_url).netloc == urlparse(url).netloc:
                scrape_page_for_rag(next_url, depth + 1, max_depth, results)

    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")

    return results


# --- CHUNKING ---

def get_chunks(text, chunk_size=300, overlap=30):
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(enc.decode(chunk))
    return chunks


# --- EMBEDDING + VECTOR DB ---
client = OpenAI()


def create_vector_db(result_blocks, index_path="faiss_index.index", meta_path="metadata.json"):
    chunks = []
    metadata = []

    for block in result_blocks:
        try:
            url = block.split('[URL]: ')[1].split('\n')[0].strip()
            content = block.split('[CONTENT]:')[1].strip()
        except Exception as e:
            print(f"Skipping malformed block: {e}")
            continue

        for chunk in get_chunks(content):
            chunks.append(chunk)
            metadata.append({
                "url": url,
                "text": chunk
            })

    print(f"Generating embeddings for {len(chunks)} chunks...")

    embeddings = []
    for chunk in tqdm(chunks, desc="Embedding chunks", unit="chunk"):
        embedding = client.embeddings.create(
            input=[chunk],
            model="text-embedding-3-small"
        ).data[0].embedding
        embeddings.append(embedding)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved FAISS index to {index_path}")
    print(f"✅ Saved metadata to {meta_path}")


def load_vector_db(index_path="RAG/faiss_index.index", meta_path="RAG/metadata.json"):
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def print_bot_response(text):
    print(Fore.CYAN + text + Style.RESET_ALL)


def extract_text_with_links(soup: BeautifulSoup, base_url: str) -> str:
    output = []

    for element in soup.find_all(['p', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        text_parts = []

        for part in element.descendants:
            if part.name == 'a' and part.get('href'):
                href = urljoin(base_url, part['href'])
                label = part.get_text(strip=True)
                text_parts.append(f"{label} ({href})")
            elif part.name is None and part.string:
                stripped = part.string.strip()
                if stripped:
                    text_parts.append(stripped)

        if text_parts:
            output.append(" ".join(text_parts))

    return "\n".join(output)
