import asyncio
import json
import logging
import os
import requests
import time
from datetime import datetime, timedelta
from utils import print_bot_response, build_rag, load_vector_db
from agents import trace, Runner, TResponseInputItem, Agent, function_tool
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
from googleapiclient.discovery import build
from google.oauth2 import service_account

load_dotenv()
logger = logging.getLogger()

SYSTEM_PROMPT = """
 You are customer service assistant helps users with customer service inquiries. It can provide information about products, services, and policies, as well as help users resolve issues and complete transactions.
 PRIORITIZE BOOKING A MEETING WITH THE USER, make it an interactive experience.
 Utilize the following functions effectively:
- **book meeting**: Use this when a user asks for a demo or meeting or to talk to customer service, get details such as email, start time/date, and timezone).
    - format timezones like the following: "America/New_York","America/Los_Angeles","Asia/Tokyo","Europe/London"
    - if no year is given in date, assume it is the current year
- **rag_search**: for general questions
 """

with open('calendar_fixed_args.json', 'r') as file:
    data = json.load(file)


SERVICE_ACCOUNT_INFO = data.get('service_account_info', {})
DELEGATED_USER = data.get('delegated_user', "")
SCOPES = ["https://www.googleapis.com/auth/calendar"]


@function_tool
def book_meeting(
    email: str, timezone: str, title: str, start_time: str
):

    try:
        credentials = (
            service_account.Credentials.from_service_account_info(
                SERVICE_ACCOUNT_INFO, scopes=SCOPES
            ).with_subject(DELEGATED_USER)
        )
        service = build("calendar", "v3", credentials=credentials)
    except Exception as e:
        logger.warning(e)

    start_time_obj: datetime = datetime.fromisoformat(start_time)
    duration_td = timedelta(minutes=30)
    end_time_obj = start_time_obj + duration_td
    end_time = end_time_obj.isoformat()

    try:
        final_event = {
            "summary": title,
            "description": "A meeting to discuss project updates.",
            "start": {
                "dateTime": start_time,
                "timeZone": timezone,
            },
            "end": {
                "dateTime": end_time,
                "timeZone": timezone,
            },
            "attendees": [
                {"email": email},
            ],
        }

        event = (
            service.events().insert(calendarId='primary', body=final_event).execute()
        )
        print("Event created: %s" % (event.get("htmlLink")))

    except Exception as e:
        logger.error(f"Error booking meeting: {e}")
        return "Failed to book meeting"

    return "Successfully booked meeting"


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


calendar_agent = Agent(
    name="Bot",
    instructions=SYSTEM_PROMPT,
    tools=[book_meeting, rag_search],
    model="gpt-4o-mini",
)


async def main():
    # calender bot logic
    input_items: list[TResponseInputItem] = []

    while True:
        user_input = input("You: ")
        if user_input == "quit":
            return
        with trace("Book A Meeting Bot"):
            start_time: float = time.time()
            input_items.append({"content": user_input, "role": "user"})
            result = await Runner.run(calendar_agent, input_items)
            print(f"getAPIBotResponse Time: {time.time() - start_time}")
            print_bot_response(f"Bot: {result.final_output}")
            input_items = result.to_input_list()


if __name__ == "__main__":
    rag_folder, url = "RAG_customer_service", "https://www.richtechrobotics.com/"
    build_rag(rag_folder, url)
    index, metadata = load_vector_db(
        index_path=f"{rag_folder}/faiss_index.index",
        meta_path=f"{rag_folder}/metadata.json",
    )
    asyncio.run(main())
