import argparse
import base64
import json
import mimetypes
import os
import requests
from importlib.metadata import version, PackageNotFoundError
from urllib.parse import urlparse
from dataclasses import dataclass, field, fields

ENV_VAR_PREFIX = "ASK2API_"
TYPE_HINTS = {
    "string": "string",
    "str": "string",
    "number": "number",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "array": "array",
    "list": "array",
    "object": "object",
    "dict": "object",
}
OPENAI_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
OPENAI_DEFAULT_MODEL = "gpt-4.1"
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-5"
ANTHROPIC_VERSION = "2023-06-01"  # Vision support available in this version and later
SYSTEM_PROMPT = """
You are a JSON API engine.

You must answer every user request as a valid API response that strictly
follows the given JSON schema.

Never return markdown, comments or extra text.
"""


@dataclass
class Config:
    api_key: str | None = field(
        default=None,
        metadata={"help": "API key (required)"},
    )
    base_url: str | None = field(
        default=None,
        metadata={"help": "Base API URL"},
    )
    model: str | None = field(
        default=None,
        metadata={"help": "Model name"},
    )
    temperature: float = field(
        default=0,
        metadata={"help": "Temperature setting"},
    )
    provider: str | None = field(
        default=None,
        metadata={"help": "API provider (openai or anthropic)"},
    )

    def __post_init__(self):
        # Default provider to openai if not specified (backward compatibility)
        if not self.provider:
            self.provider = "openai"

        # Validate provider
        if self.provider not in ["openai", "anthropic"]:
            raise ValueError(
                f"Invalid provider: {self.provider}. Must be 'openai' or 'anthropic'"
            )

        # Apply provider-specific defaults
        if self.provider == "openai":
            if not self.base_url:
                self.base_url = OPENAI_BASE_URL
            if not self.model:
                self.model = OPENAI_DEFAULT_MODEL
            self.url = f"{self.base_url}/chat/completions"
        else:  # anthropic
            if not self.base_url:
                self.base_url = ANTHROPIC_BASE_URL
            if not self.model:
                self.model = ANTHROPIC_DEFAULT_MODEL
            self.url = f"{self.base_url}/messages"

        # Validate API key
        if not self.api_key:
            raise ValueError("API key is not set!")

    @classmethod
    def get_env_vars_help(cls):
        longest = max(len(f.name) for f in fields(cls))

        def field_help(f):
            desc = f.metadata["help"]
            # Don't show defaults for api_key or fields with None defaults
            if (
                f.name == "api_key"
                or f.default is None
                or (hasattr(f.default, "default") and f.default.default is None)
            ):
                default = None
            else:
                default = getattr(cls, f.name, None)
            return "\t".join(
                [
                    f"{ENV_VAR_PREFIX}{f.name.upper():<{longest}}",
                    f"{desc} {f'(default: {default})' if default is not None else ''}",
                ]
            )

        help_text = "Environment Variables:\n" + "\n".join(
            field_help(f) for f in fields(cls)
        )

        # Add provider-specific API key info
        help_text += (
            "\n\nProvider-specific API keys (fallback if ASK2API_API_KEY not set):"
        )
        help_text += f"\n\t{'ANTHROPIC_API_KEY':<{longest}}\tAnthropic API key"
        help_text += f"\n\t{'OPENAI_API_KEY':<{longest}}\tOpenAI API key"

        return help_text

    @classmethod
    def from_env(cls):
        """Get the configuration from the environment variables."""
        # Load config from ASK2API_* environment variables
        config_dict = dict(
            filter(
                lambda x: x[1] is not None,
                {
                    name: os.getenv(ENV_VAR_PREFIX + name.upper())
                    for name in cls.__annotations__
                }.items(),
            )
        )

        # Get provider (defaults to openai in __post_init__ if not specified)
        provider = config_dict.get("provider", os.getenv("ASK2API_PROVIDER", "openai"))

        # Handle API key fallback based on provider
        if "api_key" not in config_dict or config_dict["api_key"] is None:
            # Check ASK2API_API_KEY first
            api_key = os.getenv("ASK2API_API_KEY")
            if not api_key:
                # Fallback to provider-specific key
                if provider == "anthropic":
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                else:  # openai or default
                    api_key = os.getenv("OPENAI_API_KEY")
            config_dict["api_key"] = api_key

        config_dict["provider"] = provider
        return cls(**config_dict)


def is_url(path):
    """Check if the given path is a URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def get_image_mime_type(image_path):
    """Get MIME type for an image file."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    # Fallback for common image extensions
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_map.get(ext, "image/jpeg")


def encode_image(image_path):
    """Encode image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def prepare_image_content(image_path, provider="openai"):
    """Prepare image content in OpenAI format (will be converted for Anthropic later if needed)."""
    if is_url(image_path):
        if provider == "anthropic":
            # For Anthropic with URLs, download and convert to data URL
            # since Anthropic doesn't support direct image URLs
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(image_path, headers=headers)
            response.raise_for_status()
            image_data = base64.b64encode(response.content).decode("utf-8")

            # Get mime type
            content_type = response.headers.get("content-type", "")
            if content_type.startswith("image/"):
                mime_type = content_type
            else:
                mime_type = get_image_mime_type(image_path)

            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
            }
        else:
            # OpenAI supports direct URLs
            return {"type": "image_url", "image_url": {"url": image_path}}
    else:
        # Local file - encode to base64 for both providers
        base64_image = encode_image(image_path)
        mime_type = get_image_mime_type(image_path)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
        }


def get_version():
    """Get the installed package version."""
    try:
        return version("ask2api")
    except PackageNotFoundError:
        return "dev"


def convert_example_to_schema(example, _cache=None):
    """Convert a JSON example to a JSON Schema with memoization."""
    if _cache is None:
        _cache = {}

    # Use id() for memoization key to handle nested structures
    cache_key = id(example)
    if cache_key in _cache:
        return _cache[cache_key]

    if isinstance(example, dict):
        schema = {
            "type": "object",
            "properties": {},
            "required": list(example.keys()),
            "additionalProperties": False,
        }

        for key, value in example.items():
            if isinstance(value, str):
                schema["properties"][key] = {
                    "type": TYPE_HINTS.get(value.lower(), "string")
                }
            elif isinstance(value, bool):
                schema["properties"][key] = {"type": "boolean"}
            elif isinstance(value, int):
                schema["properties"][key] = {"type": "integer"}
            elif isinstance(value, float):
                schema["properties"][key] = {"type": "number"}
            elif isinstance(value, list):
                schema["properties"][key] = {
                    "type": "array",
                    "items": (
                        convert_example_to_schema(value[0], _cache) if value else {}
                    ),
                }
            elif isinstance(value, dict):
                schema["properties"][key] = convert_example_to_schema(value, _cache)
            else:
                schema["properties"][key] = {"type": "string"}

        _cache[cache_key] = schema
        return schema

    elif isinstance(example, list):
        schema = {
            "type": "array",
            "items": convert_example_to_schema(example[0], _cache) if example else {},
        }
        _cache[cache_key] = schema
        return schema

    else:
        # Primitive types - use type() for faster checking
        type_map = {str: "string", bool: "boolean", int: "integer", float: "number"}
        schema = {"type": type_map.get(type(example), "string")}
        _cache[cache_key] = schema
        return schema


def read_text_file(path):
    """Read content from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_openai_payload(user_content, schema, config):
    """Build the payload for the OpenAI format."""
    return {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "ask2api_schema", "schema": schema},
        },
        "temperature": config.temperature,
    }


def build_openai_headers(config):
    """Build the headers for the OpenAI format."""
    return {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }


def parse_openai_response(response_json: dict) -> dict:
    """Parse OpenAI API response to extract JSON content."""
    content = response_json["choices"][0]["message"]["content"]
    return json.loads(content)


def convert_schema_to_anthropic_tool(schema: dict) -> dict:
    """Convert JSON Schema to Anthropic tool definition."""
    return {
        "name": "format_response",
        "description": "Format the API response strictly according to the provided JSON schema.",
        "input_schema": schema,
    }


def prepare_anthropic_image_content(image_path):
    """Prepare image content for Anthropic API (different format from OpenAI).

    Note: Anthropic API only supports base64-encoded images, not direct URLs.
    If a URL is provided, we download and convert it to base64.
    """
    if is_url(image_path):
        # Download the image from URL and convert to base64
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(image_path, headers=headers)
        response.raise_for_status()
        image_data = base64.b64encode(response.content).decode("utf-8")

        # Try to get mime type from response headers, fallback to guessing from URL
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("image/"):
            mime_type = content_type
        else:
            mime_type = get_image_mime_type(image_path)

        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": image_data},
        }
    else:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": get_image_mime_type(image_path),
                "data": encode_image(image_path),
            },
        }


def convert_content_for_anthropic(user_content):
    """Convert OpenAI-style multimodal content to Anthropic format."""
    anthropic_content = []
    for item in user_content:
        if item["type"] == "text":
            anthropic_content.append({"type": "text", "text": item["text"]})
        elif item["type"] == "image_url":
            # Extract image path/URL from OpenAI format
            image_url = item["image_url"]["url"]
            # Check if it's a data URL or regular URL/path
            if image_url.startswith("data:"):
                # Extract base64 data and mime type from data URL
                # Format: data:image/jpeg;base64,/9j/4AAQ...
                header, data = image_url.split(",", 1)
                mime_type = header.split(";")[0].split(":")[1]
                anthropic_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": data,
                        },
                    }
                )
            else:
                # Regular URL
                anthropic_content.append(
                    {"type": "image", "source": {"type": "url", "url": image_url}}
                )
    return anthropic_content


def build_anthropic_payload(user_content, schema, config):
    """Build the payload for the Anthropic format."""
    tool = convert_schema_to_anthropic_tool(schema)

    if isinstance(user_content, str):
        messages = [{"role": "user", "content": user_content}]
    else:
        # Convert image format from OpenAI style to Anthropic style
        messages = [
            {"role": "user", "content": convert_content_for_anthropic(user_content)}
        ]

    return {
        "model": config.model,
        "max_tokens": 4096,  # Required by Anthropic
        "system": SYSTEM_PROMPT.strip(),
        "messages": messages,
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": "format_response"},
        "temperature": config.temperature,
    }


def build_anthropic_headers(config):
    """Build the headers for the Anthropic format."""
    return {
        "x-api-key": config.api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def parse_anthropic_response(response_json: dict) -> dict:
    """Parse Anthropic API response to extract tool use result."""
    for block in response_json["content"]:
        if block["type"] == "tool_use" and block["name"] == "format_response":
            return block["input"]
    raise ValueError("No valid response found in Anthropic output")


def generate_api_response(
    user_content: str | list[dict],
    schema: dict,
    config: Config,
) -> dict:
    """Generate an API response using the configured provider."""
    if config.provider == "anthropic":
        headers = build_anthropic_headers(config)
        payload = build_anthropic_payload(user_content, schema, config)
        response = requests.post(config.url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

        # Debug: Print response if there's an error
        if "error" in response_json:
            raise ValueError(f"Anthropic API error: {response_json['error']}")

        return parse_anthropic_response(response_json)
    else:  # openai
        headers = build_openai_headers(config)
        payload = build_openai_payload(user_content, schema, config)
        response = requests.post(config.url, headers=headers, json=payload)
        response.raise_for_status()
        return parse_openai_response(response.json())


def main():
    parser = argparse.ArgumentParser(
        description="Ask a language model to return a JSON object that strictly follows a provided JSON schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=Config.get_env_vars_help(),
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "-p",
        "--prompt",
        help="Natural language prompt",
    )
    prompt_group.add_argument(
        "-pf",
        "--prompt-file",
        help="Path to text file containing the prompt",
    )
    schema_group = parser.add_mutually_exclusive_group(required=True)
    schema_group.add_argument(
        "-e",
        "--example",
        help='JSON example as a string (e.g., \'{"country": "France", "city": "Paris"}\')',
    )
    schema_group.add_argument(
        "-ef",
        "--example-file",
        help="Path to text file containing JSON example",
    )
    schema_group.add_argument(
        "-sf",
        "--schema-file",
        help="Path to JSON schema file",
    )
    parser.add_argument(
        "-i",
        "--image",
        help="Path to image file or image URL",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {get_version()}",
    )
    args = parser.parse_args()

    # Get prompt from file or argument
    prompt = read_text_file(args.prompt_file) if args.prompt_file else args.prompt

    # Load schema from file or parse from string
    if args.schema_file:
        with open(args.schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)
    else:
        example_str = (
            read_text_file(args.example_file) if args.example_file else args.example
        )
        example = json.loads(example_str)
        schema = convert_example_to_schema(example)

    config = Config.from_env()

    # Build user message content
    if args.image:
        # Multimodal content: text + image
        user_content = [
            {"type": "text", "text": prompt},
            prepare_image_content(args.image, config.provider),
        ]
    else:
        # Text-only content
        user_content = prompt

    result = generate_api_response(user_content, schema, config)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
