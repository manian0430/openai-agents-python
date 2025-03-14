import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents import set_default_openai_client, set_default_openai_api

# Load environment variables from .env file
load_dotenv()

# Google Gemini API configuration
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.0-pro-exp-02-05"

if not API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file or environment variables")

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

@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."

async def main():
    # Create an agent using the Google Gemini model
    agent = Agent(
        name="Gemini Assistant",
        instructions="You are a helpful assistant powered by Google's Gemini model.",
        model=MODEL_NAME,  # Use the Gemini model name
        tools=[get_weather],
    )

    # Run the agent
    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main()) 