import asyncio
import os
import pathlib
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import Agent, Runner, set_default_openai_client, set_default_openai_api, set_tracing_disabled

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


async def main():
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=MODEL_NAME,  # Use the Gemini model name
    )

    result = await Runner.run(agent, "Tell me about recursion in programming.")
    print(result.final_output)
    # Function calls itself,
    # Looping in smaller pieces,
    # Endless by design.


if __name__ == "__main__":
    asyncio.run(main())
