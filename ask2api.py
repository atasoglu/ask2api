import argparse
import json
import os
import requests

API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", required=True)
    parser.add_argument("-sf", "--schema-file", required=True)
    args = parser.parse_args()

    with open(args.schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)

    system_prompt = """
    You are a JSON API engine.

    You must answer every user request as a valid API response that strictly
    follows the given JSON schema.

    Never return markdown, comments or extra text.
    """

    payload = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "ask2api_schema", "schema": schema},
        },
        "temperature": 0,
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    r = requests.post(OPENAI_URL, headers=headers, json=payload)
    r.raise_for_status()

    result = r.json()["choices"][0]["message"]["content"]
    parsed_result = json.loads(result)
    print(json.dumps(parsed_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
