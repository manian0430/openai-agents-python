from __future__ import annotations

import asyncio
import os
import pathlib
from dotenv import load_dotenv

from openai import AsyncOpenAI

from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    function_tool,
    set_tracing_disabled,
)

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


"""This example creates a custom ModelProvider for Google Gemini that can be used with specific run
configurations, without affecting the global settings. Steps:
1. Create a custom OpenAI client that connects to Google's API.
2. Create a ModelProvider that uses the custom client.
3. Use the ModelProvider in calls to Runner.run().

Note: We disable tracing since it might require the regular OpenAI API.
"""
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)


class GoogleGeminiProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(model=model_name or MODEL_NAME, openai_client=client)


GEMINI_PROVIDER = GoogleGeminiProvider()
print("Created Google Gemini provider")


@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


async def main():
    agent = Agent(
        name="Gemini Assistant", 
        instructions="You only respond in haikus.", 
        tools=[get_weather]
    )

    # This will use the Google Gemini provider
    result = await Runner.run(
        agent,
        "What's the weather in Tokyo?",
        run_config=RunConfig(model_provider=GEMINI_PROVIDER),
    )
    print(result.final_output)

    # If you uncomment this, it will try to use OpenAI directly, not Gemini
    # You'll need an OPENAI_API_KEY environment variable set for this to work
    # result = await Runner.run(
    #     agent,
    #     "What's the weather in Tokyo?",
    # )
    # print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
