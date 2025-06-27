import asyncio
import requests
from io import BytesIO
from openai import OpenAI
from agents import Agent, FileSearchTool, Runner, WebSearchTool

client = OpenAI()


def create_file(client, file_path):
    if file_path.startswith("http://") or file_path.startswith("https://"):
        # Download the file content from the URL
        response = requests.get(file_path)
        file_content = BytesIO(response.content)
        file_name = file_path.split("/")[-1]
        file_tuple = (file_name, file_content)
        result = client.files.create(
            file=file_tuple,
            purpose="assistants"
        )
    else:
        # Handle local file path
        with open(file_path, "rb") as file_content:
            result = client.files.create(
                file=file_content,
                purpose="assistants"
            )
    return result.id


async def main():
    # Replace with your own file path or URL
    file_id = create_file(
        client, "richtech_data.txt")
    # create vector store
    vector_store = client.vector_stores.create(
        name="knowledge_base"
    )
    # add file to store
    client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file_id
    )

    result = client.vector_stores.files.list(
        vector_store_id=vector_store.id
    )

    agent = Agent(
        name="Assistant",
        tools=[
            WebSearchTool(),
            FileSearchTool(
                max_num_results=1,
                vector_store_ids=[f"{vector_store.id}"],
            ),
        ],
    )
    result = await Runner.run(agent, "What type of robots do you sell?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
