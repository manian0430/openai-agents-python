# Custom LLM Providers & Google Gemini

The examples in this directory demonstrate how to use non-OpenAI LLM providers with the Agents SDK, particularly Google's Gemini models.

## Google Gemini Examples

These examples have been configured to use Google's Gemini models via their OpenAI-compatible API endpoint.

To run these examples, first create a `.env` file in the project root with your Google API key:

```
GOOGLE_API_KEY=your_google_api_key_here
```

Then run any of the examples:

```
python examples/model_providers/custom_example_agent.py
python examples/model_providers/custom_example_global.py
python examples/model_providers/custom_example_provider.py
```

Each example demonstrates a different approach to using Google Gemini:

1. `custom_example_agent.py` - Configures a specific agent to use Gemini without changing global settings
2. `custom_example_global.py` - Sets Gemini as the default LLM for all agents
3. `custom_example_provider.py` - Creates a custom provider that can be used with specific run configurations

## Using with Other LLM Providers

If you want to use a different LLM provider that offers an OpenAI-compatible API, simply modify the following variables in the examples:

```python
BASE_URL = "your_provider_base_url"
API_KEY = "your_provider_api_key"
MODEL_NAME = "your_provider_model_name"
```

## Generic Custom Provider Examples

To run the generic examples, first set a base URL, API key and model:

```bash
export EXAMPLE_BASE_URL="..."
export EXAMPLE_API_KEY="..."
export EXAMPLE_MODEL_NAME="..."
```

Then run the examples, e.g.:

```
python examples/model_providers/custom_example_provider.py

Loops within themselves,
Function calls its own being,
Depth without ending.
```
