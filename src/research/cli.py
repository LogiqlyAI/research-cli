import os
import sys
import json
import textwrap
import argparse
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
import anthropic

SYSTEM_PROMPT = "Provide a concise, balanced answer. If the question is ambiguous or lacks context, state that clearly. Distinguish between established facts and your own inference. If you don't know, say so. Avoid speculation and do not fabricate information. Use clear, simple language suitable for a general audience."

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

DEFAULT_MAX_TOKENS = 4096

class Report(BaseModel):
    """Structured output from Claude"""
    summary: str = Field(description="2-3 sentence summary, highlighting the core answer.")
    key_points: list[str] = Field(description="5-7 key insights, each a clear sentence.")
    follow_ups: list[str] = Field(description="4-6 thought-provoking follow-up questions.")

class TruncatedResponseError(Exception):
    """Raised when the model's response is cut off by the token limit."""
    pass

class ConfigError(Exception):
    """Raised when configuration (env, args, etc.) is invalid."""
    pass

@dataclass
class Config:
    """Resolved configuration for the research tool."""
    model: str
    max_tokens: int

def _parse_int_gracefully(value: str, name: str) -> int:
    """Parse an integer from a string, raising ConfigError on failure."""
    try:
        return int(value)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got: {value!r}")
    
def resolve_config(args: argparse.Namespace) -> Config:
    """Resolve configuration from CLI flags, environment variables, and defaults.
    
    Precedence (highest to lowest):
        1. CLI flags (--model, --max-tokens) - only if explicitly provided
        2. Environment variables (RESEARCH_MODEL, RESEARCH_MAX_TOKENS)
        3. Built-in defaults (DEFAULT_MODEL, DEFAULT_MAX_TOKENS)

    Raises:
        ConfigError: if any configuration is invalid (e.g., missing API key, bad max_tokens, etc.)
    """
    load_dotenv()  # Idempotent - safe to call multiple times

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ConfigError("ANTHROPIC_API_KEY not found. Set it with: export ANTHROPIC_API_KEY='your-key' or create a .env file.")

    model = (
        args.model
        if args.model is not None 
        else os.getenv("RESEARCH_MODEL") or DEFAULT_MODEL
    )

    env_max_tokens = os.getenv("RESEARCH_MAX_TOKENS")
    if args.max_tokens is not None:
        max_tokens = args.max_tokens
    elif env_max_tokens is not None:
        max_tokens = _parse_int_gracefully(env_max_tokens, "RESEARCH_MAX_TOKENS")
    else:
        max_tokens = DEFAULT_MAX_TOKENS

    if max_tokens < 1:
        raise ConfigError(f"max-tokens must be at least 1, got: {max_tokens}")

    return Config(model=model, max_tokens=max_tokens)

def get_report(question: str, config: Config) -> Report:
    """Ask Claude and return a validated Report
    
       Raises:
          ValueError: if the question is empty.
          TruncatedResponseError: if the response was cut off.
          anthropic.APIError: for API-level failures.
          ValidationError: if the model's output doesn't match the Report schema. 
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=config.model,
        system=SYSTEM_PROMPT,
        max_tokens=config.max_tokens,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        output_format=Report,
    )

    if response.stop_reason == "max_tokens":
        raise TruncatedResponseError(f"Response truncated at {config.max_tokens} tokens. Increase --max-tokens.")

    return response.parsed_output

def render(report: Report) -> None:
    """Print a human-readable version of the report to stdout."""
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(textwrap.fill(report.summary, width=80))

    print("\n" + "=" * 80)
    print("KEY POINTS")
    print("=" * 80)
    for i, pt in enumerate(report.key_points, 1):
        indent = ' ' * (len(str(i)) + 2)
        print(f"{i}. {textwrap.fill(pt, width=76, subsequent_indent=indent)}")

    print("\n" + "=" * 80)
    print("FOLLOW-UP QUESTIONS")
    print("=" * 80)
    for i, fu in enumerate(report.follow_ups, 1):
        indent = ' ' * (len(str(i)) + 2)
        print(f"{i}. {textwrap.fill(fu, width=76, subsequent_indent=indent)}")
    print("=" * 80)

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Ask Claude a question and get a structured report."
    )
    parser.add_argument(
        "question",
        help="The question to ask.",
        nargs="?"
    )
    parser.add_argument(
        "--model",
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help=f"Maximum tokens for the response (default: {DEFAULT_MAX_TOKENS})"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON (for piping to jq, etc.)"
    )

    args = parser.parse_args()

    if not args.question or not args.question.strip():
        parser.print_help()
        sys.exit(1)

    try:
        config = resolve_config(args)
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        report = get_report(question=args.question, config=config)
    except ValidationError as e:
        print(f"Schema validation error – the model return unexpected structure:\n{e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Input error: {e}", file=sys.stderr)
        sys.exit(1)
    except TruncatedResponseError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIError as e:
        print(f"API error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(report.model_dump(), indent=2))
    else:
        render(report)

if __name__ == "__main__":
    main()