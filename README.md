# ask2api

[![CI](https://github.com/atasoglu/ask2api/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/atasoglu/ask2api/actions/workflows/pre-commit.yml)
[![PyPI version](https://img.shields.io/pypi/v/ask2api)](https://pypi.org/project/ask2api/)
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
- Vision modality support
- Designed for automation pipelines and AI-driven backend workflows

## Installation

```bash
pip install ask2api
```

Set your API key:

```bash
export ASK2API_API_KEY="your_api_key"
# Or you can pass OpenAI key
# export OPENAI_API_KEY="your_api_key"
```

## Usage

### Text-only prompts

Instead of asking:

> *"Where is the capital of France?"*

and receiving free-form text, you can do this:

```bash
ask2api -p "Where is the capital of France?" -sf schema.json
```

Or pass an example directly without a schema file:

```bash
ask2api -p "Where is the capital of France?" -e '{"country": "string", "city": "string"}'
```

And get a structured API response:

```json
{
  "country": "France",
  "city": "Paris"
}
```

<details>

<summary>For more complex structures with different data types:</summary>

```bash
ask2api -p "Analyze carbon element" -e '{
  "symbol": "element symbol",
  "atomic_number": 1234,
  "atomic_weight": 12.34,
  "is_metal": true,
  "isotopes": ["name of the isotope"],
  "properties": {
    "melting_point": 1234.5,
    "boiling_point": 2345.6,
    "magnetic": true
  }
}'
```

Output:

```
{
  "symbol": "C",
  "atomic_number": 6,
  "atomic_weight": 12.011,
  "is_metal": false,
  "isotopes": [
    "C-12",
    "C-13",
    "C-14"
  ],
  "properties": {
    "melting_point": 3550,
    "boiling_point": 4827,
    "magnetic": false
  }
}
```

</details>

### Vision modality

You can also analyze images and get structured JSON responses:

```bash
ask2api -p "Where is this place?" -sf schema.json -i https://upload.wikimedia.org/wikipedia/commons/6/64/Lesdeuxmagots.jpg
```

## How it works

1. You define the desired output structure using a JSON Schema.
2. The schema is passed to the model using OpenAI's `json_schema` structured output format.
3. The system prompt enforces strict JSON-only responses.
4. For vision tasks, images are automatically encoded (base64 for local files) or passed as URLs.
5. The CLI prints the API-ready JSON output.

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
