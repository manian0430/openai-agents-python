from __future__ import annotations as _annotations

import asyncio
import random
import uuid
import os
import pathlib
from dotenv import load_dotenv

# Load environment variables from the .env file in the project root
# First, determine the root directory (1 level up from this file)
current_dir = pathlib.Path(__file__).parent.absolute()
root_dir = current_dir.parent.parent
dotenv_path = root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Google Gemini API configuration
from openai import AsyncOpenAI

from agents import (
    Agent,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    handoff,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
    trace,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

from pydantic import BaseModel

# Google Gemini API configuration
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError(f"Please set GOOGLE_API_KEY in your .env file at {dotenv_path}")

MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-pro")

# Create custom client and set up configuration
client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

# Create a custom provider (alternative approach)
class GoogleGeminiProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> OpenAIChatCompletionsModel:
        return OpenAIChatCompletionsModel(model=model_name or MODEL_NAME, openai_client=client)

gemini_provider = GoogleGeminiProvider()
runner_config = RunConfig(model_provider=gemini_provider)

### CONTEXT


class AirlineAgentContext(BaseModel):
    passenger_name: str | None = None
    confirmation_number: str | None = None
    seat_number: str | None = None
    flight_number: str | None = None


### HOOKS


async def on_seat_booking_handoff(context: RunContextWrapper[AirlineAgentContext]) -> None:
    flight_number = f"FLT-{random.randint(100, 999)}"
    context.context.flight_number = flight_number


### AGENTS

faq_agent = Agent[AirlineAgentContext](
    name="FAQ Agent",
    model=MODEL_NAME,
    handoff_description="A helpful agent that can answer questions about the airline.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an FAQ agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    
    Here are some facts about our airline that you can use to answer customer questions:
    - You are allowed to bring one bag on the plane. It must be under 50 pounds and 22 inches x 14 inches x 9 inches.
    - There are 120 seats on the plane. There are 22 business class seats and 98 economy seats.
    - Exit rows are rows 4 and 16.
    - Rows 5-8 are Economy Plus, with extra legroom.
    - We have free wifi on the plane, join Airline-Wifi.
    
    # Routine
    1. Identify the last question asked by the customer.
    2. Answer the question using the facts provided above.
    3. If you cannot answer the question, transfer back to the triage agent.""",
)

seat_booking_agent = Agent[AirlineAgentContext](
    name="Seat Booking Agent",
    model=MODEL_NAME,
    handoff_description="A helpful agent that can update a seat on a flight.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a seat booking agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    
    # Routine
    1. Ask for their confirmation number.
    2. Ask the customer what their desired seat number is.
    3. After the customer provides both pieces of information, confirm their new seat assignment.
    
    When the customer provides information, update your knowledge about them:
    - When they give a confirmation number, acknowledge receipt of a valid confirmation number.
    - When they give a seat number, confirm that the seat is available and assigned to them.
    
    If the customer asks a question that is not related to the routine, transfer back to the triage agent.""",
)

triage_agent = Agent[AirlineAgentContext](
    name="Triage Agent",
    model=MODEL_NAME,
    handoff_description="A triage agent that can delegate a customer's request to the appropriate agent.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} "
        "You are a helpful triaging agent for an airline customer service system. You should:"
        "\n1. Determine the customer's needs."
        "\n2. For general questions about baggage, seats, or wifi, transfer to the FAQ Agent."
        "\n3. For seat change requests, transfer to the Seat Booking Agent."
        "\n4. For anything else, try to help directly or explain which requests you can handle."
    ),
    handoffs=[
        faq_agent,
        handoff(agent=seat_booking_agent, on_handoff=on_seat_booking_handoff),
    ],
)

faq_agent.handoffs.append(triage_agent)
seat_booking_agent.handoffs.append(triage_agent)


### RUN


async def main():
    # Print configuration for debugging
    print(f"Using Google Gemini API with model: {MODEL_NAME}")
    
    current_agent: Agent[AirlineAgentContext] = triage_agent
    input_items: list[TResponseInputItem] = []
    context = AirlineAgentContext()

    # Normally, each input from the user would be an API request to your app, and you can wrap the request in a trace()
    # Here, we'll just use a random UUID for the conversation ID
    conversation_id = uuid.uuid4().hex[:16]

    print("Welcome to Airline Customer Service! How can I help you today?")
    print("(Type 'exit' to quit)")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Thank you for using Airline Customer Service. Goodbye!")
            break

        with trace("Customer service", group_id=conversation_id):
            input_items.append({"content": user_input, "role": "user"})
            result = await Runner.run(current_agent, input_items, context=context, run_config=runner_config)

            for new_item in result.new_items:
                agent_name = new_item.agent.name
                if isinstance(new_item, MessageOutputItem):
                    print(f"{agent_name}: {ItemHelpers.text_message_output(new_item)}")
                elif isinstance(new_item, HandoffOutputItem):
                    print(
                        f"[System]: Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}"
                    )
                else:
                    print(f"[System]: {agent_name}: {new_item.__class__.__name__}")
            
            # Print context information for debugging
            if context.confirmation_number or context.seat_number or context.flight_number:
                print(f"[Debug] Context: {context}")
            
            input_items = result.to_input_list()
            current_agent = result.last_agent


if __name__ == "__main__":
    asyncio.run(main())
