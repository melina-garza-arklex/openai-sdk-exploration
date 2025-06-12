import asyncio
from utils import print_bot_response, build_rag, load_vector_db
from agents import trace, Runner, TResponseInputItem, Agent, function_tool
from openai import OpenAI
import numpy as np

SYSTEM_PROMPT = """
 You are customer service assistant helps users with customer service inquiries. It can provide information about products, services, and policies, as well as help users resolve issues and complete transactions.
 Help the customer either: 1. Search for products , 2. Get product info
 Utilize the following functions effectively:

- **”find_product”**: Search for specific products based on customer inquiries.
— **”get_products”**: Access a comprehensive list of products to offer additional options.
 """

client = OpenAI()


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
    tools=[rag_search],
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
