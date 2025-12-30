# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.3] - 2025-12-30

### Added
- **README:** Note warning that some API providers may require base64-encoded images (local files) instead of image URLs; added example commands (curl/wget) to download images and use local files with `ask2api`.

## [1.1.2] - 2025-12-28

### Fixed
- Added `license` field to pyproject.toml for proper pip metadata display

## [1.1.1] - 2025-12-28

### Added
- `generate_api_response` function to streamline API response generation using OpenAI format
- `build_payload` function for better organization of request payload construction
- `build_headers` function for better organization of request headers construction
- System prompt constant variable for improved maintainability

### Changed
- Refactored main function to utilize new response generation logic for enhanced code clarity
- Improved code organization and maintainability through function extraction

## [1.1.0] - 2025-12-28

### Added
- File input support for prompts: `-pf` / `--prompt-file` option to read prompts from text files
- File input support for examples: `-ef` / `--example-file` option to read JSON examples from text files
- `read_text_file` utility function for reading file contents
- Dynamic help text generation for environment variables in CLI
- Metadata documentation for Config dataclass fields

### Changed
- Prompt input now supports both direct string (`-p`) and file input (`-pf`) as mutually exclusive options
- Example input now supports both direct string (`-e`) and file input (`-ef`) as mutually exclusive options
- Enhanced Config dataclass with field metadata for better documentation
- Improved CLI help output with `get_env_vars_help` method

## [1.0.0] - 2025-12-27

### Added
- Example input support: `-e` option to accept JSON examples directly without schema files
- Dynamic schema generation from JSON examples via `convert_example_to_schema` function
- Support for complex nested structures with automatic type inference
- Type hints mapping for common type names (string, int, float, bool, array, object)

### Changed
- Schema input is now optional when using `-e` flag
- Made `-sf` and `-e` mutually exclusive options

## [0.3.0] - 2025-12-27

### Changed
- Refactored API configuration to use `Config` dataclass for better maintainability
- Replaced hardcoded API settings with configurable values
- Added environment variable support for API configuration:
  - `ASK2API_BASE_URL` - Override the base API URL (default: `https://api.openai.com/v1`)
  - `ASK2API_MODEL` - Override the model name (default: `gpt-4.1`)
  - `ASK2API_TEMPERATURE` - Override the temperature setting (default: `0`)

## [0.2.1] - 2025-02-27

### Added
- CLI version flag (`--version` or `-v`) to display installed package version

## [0.2.0] - 2025-02-27

### Added
- Vision modality support: analyze images and get structured JSON responses
- Support for image files (local) and image URLs
- Automatic base64 encoding for local image files
- Image MIME type detection and handling

## [0.1.0] - 2025-12-27

### Added
- Initial release
- CLI tool to convert natural language prompts to structured JSON API responses
- JSON Schema validation support
- OpenAI integration with structured output format
- Minimal dependencies and CLI-first design
