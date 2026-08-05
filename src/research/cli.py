import sys
import textwrap

from pydantic import BaseModel, Field, ValidationError
import anthropic

SYSTEM_PROMPT = "Provide a concise, balanced answer. If the question is ambiguous or lacks context, state that clearly. Distinguish between established facts and your own inference. If you don't know, say so. Avoid speculation and do not fabricate information. Use clear, simple language suitable for a general audience."

MODEL = "claude-haiku-4-5-20251001"

MAX_TOKENS = 4096

class Report(BaseModel):
    summary: str = Field(description="2-3 sentence summary, highlighting the core answer.")
    key_points: list[str] = Field(description="5-7 key insights, each a clear sentence.")
    follow_ups: list[str] = Field(description="4-6 thought-provoking follow-up questions.")

def get_report(question: str) -> Report:
    """Ask Claude and return a validated Report"""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    client = anthropic.Anthropic()

    response = client.messages.parse(
        model=MODEL,
        system=SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        output_format=Report,
    )

    if response.stop_reason == "max_tokens":
        raise RuntimeError(f"Response truncated at {MAX_TOKENS} tokens. Increase MAX_TOKENS.")

    return response.parsed_output

def render(report: Report) -> None:
    """Pretty-print the report."""
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

if __name__ == "__main__":
    try:
        question = input("Enter your question: ")
    except EOFError:
        print("No input provided. Exiting.", file=sys.stderr)
        sys.exit(1)

    try:
        report = get_report(question)
    except ValueError as e:
        print(f"Input error: {e}", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIError as e:
        print(f"API error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    render(report)