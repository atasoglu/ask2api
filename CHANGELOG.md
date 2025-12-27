# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
