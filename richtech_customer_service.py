import asyncio
import os
import json
from openai import OpenAI
import numpy as np
from agents import (
    Agent,
    Runner,
    TResponseInputItem,
    trace,
    function_tool,
    WebSearchTool,
)
from utils import (
    print_bot_response,
    build_rag,
    load_vector_db,
)

from openai import OpenAI
import faiss
import numpy as np
import json

SYSTEM_PROMPT = """
You are a helpful assistant for Richtech Robotics customer service.

Company Overview:
- Richtech Robotics offers worker (ADAM, Scorpion), delivery (Matradee Plus, Titan, Medbot), and cleaning (DUST-E S, DUST-E MX) robots.
- ADAM is a robot bartender that prepares tea, coffee, and cocktails. Available for rental or purchase.
- Robots are intended for business use only (not home use).
- ClouTea in Las Vegas is the world's first robot milk tea shop, operated by ADAM.
- Delivery: ADAM = 2 weeks, Delivery robots = 1 month, Cleaning robots = 2 months.

If RAGSearch does not provide a helpful answer, try the WebSearchTool to find up-to-date or missing information.
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


customer_service_agent = Agent(
    name="Bot",
    instructions=SYSTEM_PROMPT,
    tools=[rag_search, WebSearchTool()],
    model="gpt-4o-mini",
)


async def main():
    # customer service bot logic
    input_items: list[TResponseInputItem] = []

    while True:
        user_input = input("You: ")
        if user_input == "quit":
            return
        with trace("Customer service"):
            input_items.append({"content": user_input, "role": "user"})
            result = await Runner.run(customer_service_agent, input_items)
            print_bot_response(f"Bot: {result.final_output}")
            input_items = result.to_input_list()


if __name__ == "__main__":
    rag_folder, url = "RAG_customer_service", "https://www.richtechrobotics.com/"
    build_rag(rag_folder, url)
    index, metadata = load_vector_db(
        index_path=f"{rag_folder}/faiss_index.index", meta_path=f"{rag_folder}/metadata.json")
    asyncio.run(main())
