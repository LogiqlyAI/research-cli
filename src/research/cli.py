import os
import sys
import json
import textwrap
import argparse

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

def get_report(question: str, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> Report:
    """Ask Claude and return a validated Report
    
       Raises:
          ValueError: if the question is empty.
          TruncatedResponseError: if the response was cut off.
          anthropic.APIError: for API-level failures.
          ValidationError: if the model's output doesn't match the Report schema. 
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        system=SYSTEM_PROMPT,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        output_format=Report,
    )

    if response.stop_reason == "max_tokens":
        raise TruncatedResponseError(f"Response truncated at {max_tokens} tokens. Increase --max-tokens.")

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
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment or .env file. Set it with: export ANTHROPIC_API_KEY='your-key' or create a .env file.", file=sys.stderr)
        sys.exit(1)

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
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens for the response (default: {DEFAULT_MAX_TOKENS})"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON (for piping to jq, etc.)"
    )

    args = parser.parse_args()

    if not args.question:
        parser.print_help()
        sys.exit(1)

    if args.max_tokens < 1:
        print("Error: --max-tokens must be at least 1.", file=sys.stderr)
        sys.exit(1)

    try:
        report = get_report(args.question, model=args.model, max_tokens=args.max_tokens)
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