import asyncio
import random
import os
import pathlib
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import Any

from pydantic import BaseModel

from agents import Agent, AgentHooks, RunContextWrapper, Runner, Tool, function_tool
from agents import set_default_openai_client, set_default_openai_api, set_tracing_disabled

# Load environment variables from the .env file in the project root
# First, determine the root directory (2 levels up from this file)
current_dir = pathlib.Path(__file__).parent.absolute()
root_dir = current_dir.parent.parent
dotenv_path = root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Google Gemini API configuration
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.0-pro-exp-02-05"

if not API_KEY:
    raise ValueError(f"Please set GOOGLE_API_KEY in your .env file at {dotenv_path}")

# Configure the AsyncOpenAI client to use Google's Gemini API
client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)
    
# Configure the Agents SDK to use the custom client
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
    
# Disable tracing as it might need the regular OpenAI API
set_tracing_disabled(disabled=True)
print("Using Google Gemini API")


class CustomAgentHooks(AgentHooks):
    def __init__(self, display_name: str):
        self.event_counter = 0
        self.display_name = display_name

    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        print(f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} started")

    async def on_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} ended with output {output}"
        )

    async def on_handoff(self, context: RunContextWrapper, agent: Agent, source: Agent) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {source.name} handed off to {agent.name}"
        )

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} started tool {tool.name}"
        )

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} ended tool {tool.name} with result {result}"
        )


###


@function_tool
def random_number(max: int) -> int:
    """
    Generate a random number up to the provided maximum.
    """
    return random.randint(0, max)


@function_tool
def multiply_by_two(x: int) -> int:
    """Simple multiplication by two."""
    return x * 2


class FinalResult(BaseModel):
    number: int


multiply_agent = Agent(
    name="Multiply Agent",
    instructions="Multiply the number by 2 and then return the final result as a number.",
    tools=[multiply_by_two],
    hooks=CustomAgentHooks(display_name="Multiply Agent"),
    model=MODEL_NAME,
)

start_agent = Agent(
    name="Start Agent",
    instructions="Generate a random number. If it's even, stop. If it's odd, hand off to the multipler agent.",
    tools=[random_number],
    handoffs=[multiply_agent],
    hooks=CustomAgentHooks(display_name="Start Agent"),
    model=MODEL_NAME,
)


async def main() -> None:
    user_input = input("Enter a max number: ")
    await Runner.run(
        start_agent,
        input=f"Generate a random number between 0 and {user_input}.",
    )

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
"""
$ python examples/basic/agent_lifecycle_example.py

Enter a max number: 250
### (Start Agent) 1: Agent Start Agent started
### (Start Agent) 2: Agent Start Agent started tool random_number
### (Start Agent) 3: Agent Start Agent ended tool random_number with result 37
### (Start Agent) 4: Agent Start Agent started
### (Start Agent) 5: Agent Start Agent handed off to Multiply Agent
### (Multiply Agent) 1: Agent Multiply Agent started
### (Multiply Agent) 2: Agent Multiply Agent started tool multiply_by_two
### (Multiply Agent) 3: Agent Multiply Agent ended tool multiply_by_two with result 74
### (Multiply Agent) 4: Agent Multiply Agent started
### (Multiply Agent) 5: Agent Multiply Agent ended with output number=74
Done!
"""
