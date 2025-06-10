import asyncio
from colorama import Fore, Style


from agents import (
    Agent,
    ItemHelpers,
    MessageOutputItem,
    Runner,
    TResponseInputItem,
    WebSearchTool,
    trace,
)


def print_bot_response(text):
    print(Fore.CYAN + text + Style.RESET_ALL)


customer_service_agent = Agent(
    name="Bot",
    instructions="The customer service assistant helps users with customer service inquiries. It can provide information about products, services, and policies, as well as help users resolve issues and complete transactions. Company Richtech Robotics's. Headquarter is in Las Vegas; the other office is in Austin. Richtech Robotics provide worker robots (ADAM, ARM, ACE), delivery robots (Matradee, Matradee X, Matradee L, Richie), cleaning robots (DUST-E SX, DUST-E MX) and multipurpose robots (skylark). Their products are intended for business purposes, but not for home purpose; the ADAM robot is available for purchase and rental for multiple purposes. This robot bartender makes tea, coffee and cocktails. Richtech Robotics also operate the world's first robot milk tea shop, ClouTea, in Las Vegas (www.cloutea.com), where all milk tea beverages are prepared by the ADAM robot. The delivery time will be one month for the delivery robot, 2 weeks for standard ADAM, and two months for commercial cleaning robot. ",
    tools=[WebSearchTool(
        user_location={"type": "approximate", "city": "New York"})],
    model="gpt-4o-mini"
)


async def main():
    input_items: list[TResponseInputItem] = []

    while True:
        user_input = input("You: ")
        if user_input == 'quit':
            return
        with trace("Customer service"):
            input_items.append({"content": user_input, "role": "user"})
            result = await Runner.run(customer_service_agent, input_items)

            for new_item in result.new_items:
                agent_name = new_item.agent.name
                if isinstance(new_item, MessageOutputItem):
                    print_bot_response(
                        f"{agent_name}: {ItemHelpers.text_message_output(new_item)}")
            input_items = result.to_input_list()


if __name__ == "__main__":
    asyncio.run(main())
