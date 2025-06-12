import asyncio
import os
import requests
from utils import print_bot_response, build_rag, load_vector_db
from agents import trace, Runner, TResponseInputItem, Agent, function_tool
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
load_dotenv()

SHOPIFY_URL = os.getenv("SHOPIFY_URL")
SHOPIFY_ADMIN_TOKEN = os.getenv("SHOPIFY_ADMIN_TOKEN")

SYSTEM_PROMPT = """
 You are customer service assistant helps users with customer service inquiries. It can provide information about products, services, and policies, as well as help users resolve issues and complete transactions.
 Help the customer either: 
 1. Search for products 
 2. Get product info
 Utilize the following functions effectively:

- **search_products**: Use this when a user asks for product recommendations or mentions a product type (e.g. "recommend hats", "looking for shoes").
— **”get_products”**: Access a comprehensive list of products to offer additional options.
 """

client = OpenAI()


@function_tool
def search_products(product_search: str):
    url = f"{SHOPIFY_URL}/admin/api/2025-04/graphql.json"

    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": SHOPIFY_ADMIN_TOKEN
    }

    query = f"""
    query SearchProducts {{
    products(first: 3, query: "{product_search}") {{
        nodes {{
        id
        title
        handle
        description
        totalInventory
        onlineStoreUrl
        images(first: 1) {{
                                edges {{
                                    node {{
                                        src
                                        altText
                                    }}
                                }}
                            }}
        }}
    }}
    }}
    """

    payload = {
        "query": query
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        products = response.json()['data']['products']['nodes']
        # create link for product
        for product in products:
            product['onlineStoreUrl'] = f"{SHOPIFY_URL}/products/{product.get('handle')}"
        return products
    else:
        print(f"❌ Error {response.status_code}: {response.text}")


@function_tool
def get_products():
    url = f"{SHOPIFY_URL}/admin/api/2025-04/graphql.json"

    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": SHOPIFY_ADMIN_TOKEN
    }

    query = """
    query GetProducts {
    products(first: 5) {
        nodes {
        id
        title
        description
        totalInventory
        onlineStoreUrl
        }
    }
    }
    """

    payload = {
        "query": query
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"❌ Error {response.status_code}: {response.text}")


@function_tool
async def rag_search(query: str) -> str:
    """Search knowledge base to answer the user's question with citations."""
    # Embed the question
    query_embedding = (
        client.embeddings.create(input=[query], model="text-embedding-3-small")
        .data[0]
        .embedding
    )

    # Search FAISS
    D, I = index.search(np.array([query_embedding], dtype="float32"), k=3)

    context_blocks = []
    for idx in I[0]:
        if idx < len(metadata):
            entry = metadata[idx]
            context_blocks.append(f"[Source: {entry['url']}]\n{entry['text']}")

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
    {SYSTEM_PROMPT}

    Use ONLY the context below and system prompt above to answer the following question.

    Context:
    {context}

    Question:
    {query}

    Answer with any relevant citations in the format [source: URL].
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

shopify_agent = Agent(
    name="Bot",
    instructions=SYSTEM_PROMPT,
    tools=[search_products, get_products, rag_search],
    model="gpt-4o-mini",
)


async def main():
    # shopify bot logic
    input_items: list[TResponseInputItem] = []

    while True:
        user_input = input("You: ")
        if user_input == "quit":
            return
        with trace("Shopify Bot"):
            input_items.append({"content": user_input, "role": "user"})
            result = await Runner.run(shopify_agent, input_items)
            print_bot_response(f"Bot: {result.final_output}")
            input_items = result.to_input_list()


if __name__ == "__main__":
    rag_folder, url = "RAG_shopify", "https://arklex-demo-store-1.myshopify.com/"
    build_rag(rag_folder, url, is_shopify=True)
    index, metadata = load_vector_db(
        index_path=f"{rag_folder}/faiss_index.index", meta_path=f"{rag_folder}/metadata.json")
    asyncio.run(main())
