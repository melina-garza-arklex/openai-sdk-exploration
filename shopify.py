import asyncio
from utils import print_bot_response
from agents import trace, Runner, TResponseInputItem, Agent

SYSTEM_PROMPT = ""

shopify_agent = Agent(
    name="Bot",
    instructions=SYSTEM_PROMPT,
    tools=[],
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
            result = await Runner.run(shopify_agent, input_items)
            print_bot_response(f"Bot: {result.final_output}")
            input_items = result.to_input_list()


if __name__ == "__main__":
    asyncio.run(main())
