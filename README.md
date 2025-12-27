# ask2api

[![CI](https://github.com/atasoglu/ask2api/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/atasoglu/ask2api/actions/workflows/pre-commit.yml)
[![PyPI version](https://badge.fury.io/py/ask2api.svg)](https://badge.fury.io/py/ask2api)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`ask2api` is a minimal Python CLI tool that turns natural language prompts into structured API-style JSON responses using LLM.

It allows you to define a JSON Schema and force the model to answer strictly in that format.

## Why ask2api?

Because LLMs are no longer just chatbots, they are also programmable API engines.

`ask2api` lets you use them that way. 🚀

Key features:

- Minimal dependencies  
- CLI first  
- Prompt → API behavior  
- No markdown, no explanations, only valid JSON  
- Designed for automation pipelines and AI-driven backend workflows

## Installation

```bash
pip install ask2api
```

Set your OpenAI key:

```bash
export OPENAI_API_KEY="your_api_key"
```

## Usage

Instead of asking:

> *“Where is the capital of France?”*

and receiving free-form text, you can do this:

```bash
ask2api -p "Where is the capital of France?" -sf schema.json
```

And get a structured API response:

```json
{
  "country": "France",
  "city": "Paris"
}
```

## How it works

1. You define the desired output structure using a JSON Schema.
2. The schema is passed to the model using OpenAI’s `json_schema` structured output format.
3. The system prompt enforces strict JSON-only responses.
4. The CLI prints the API-ready JSON output.

The model is treated as a deterministic API function.

## Example schema

Create a file named `schema.json`:

```json
{
  "type": "object",
  "properties": {
    "country": { "type": "string" },
    "city": { "type": "string" }
  },
  "required": ["country", "city"]
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT
