import argparse
import base64
import json
import mimetypes
import os
import requests
from importlib.metadata import version, PackageNotFoundError
from urllib.parse import urlparse
from dataclasses import dataclass

API_KEY = os.getenv("OPENAI_API_KEY")
ENV_VAR_PREFIX = "ASK2API_"


@dataclass
class Config:
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1"
    temperature: float = 0

    def __post_init__(self):
        self.openai_url = f"{self.base_url}/chat/completions"

    @classmethod
    def from_env(cls, prefix: str = ENV_VAR_PREFIX):
        """Get the configuration from the environment variables."""
        return cls(
            **dict(
                filter(
                    lambda x: x[1] is not None,
                    dict(
                        base_url=os.getenv(f"{prefix}BASE_URL"),
                        model=os.getenv(f"{prefix}MODEL"),
                        temperature=os.getenv(f"{prefix}TEMPERATURE"),
                    ).items(),
                ),
            )
        )


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


def prepare_image_content(image_path):
    """Prepare image content for OpenAI API (either URL or base64 encoded)."""
    if is_url(image_path):
        return {"type": "image_url", "image_url": {"url": image_path}}
    else:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", required=True)
    parser.add_argument("-sf", "--schema-file", required=True)
    parser.add_argument("-i", "--image")
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {get_version()}"
    )
    args = parser.parse_args()

    with open(args.schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)

    system_prompt = """
    You are a JSON API engine.

    You must answer every user request as a valid API response that strictly
    follows the given JSON schema.

    Never return markdown, comments or extra text.
    """

    # Build user message content
    if args.image:
        # Multimodal content: text + image
        user_content = [
            {"type": "text", "text": args.prompt},
            prepare_image_content(args.image),
        ]
    else:
        # Text-only content
        user_content = args.prompt

    config = Config.from_env()

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "ask2api_schema", "schema": schema},
        },
        "temperature": config.temperature,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(config.openai_url, headers=headers, json=payload)
    r.raise_for_status()

    result = r.json()["choices"][0]["message"]["content"]
    parsed_result = json.loads(result)
    print(json.dumps(parsed_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
