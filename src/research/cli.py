import json
import re
import textwrap

import anthropic

SYSTEM_PROMPT = "Structure your response as a JSON object with summary (string), key_points (list of strings), and follow_ups (list of strings)."

MODEL = "claude-haiku-4-5-20251001"

MAX_TOKENS = 4096

question = input("Enter your question: ")

def ask(question: str) -> anthropic.Message:

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    client = anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        system=SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
    )
    print(f"Stop reason: {response.stop_reason}")
    return response

ai_output = ask(question)

for block in ai_output.content:
    if block.type == "text":
        cleaned_response = re.sub(r'^```json\s*|```$', '', block.text.strip(), flags=re.IGNORECASE)
        data = json.loads(cleaned_response)
        break

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(textwrap.fill(data["summary"], width=80))

print("\n" + "=" * 80)
print("KEY POINTS")
print("=" * 80)
for i, pt in enumerate(data["key_points"], 1):
    print(f"{i}. {textwrap.fill(pt, width=76, subsequent_indent=' ' * (len(str(i)) + 2))}")

print("\n" + "=" * 80)
print("FOLLOW-UP QUESTIONS")
print("=" * 80)
for i, fu in enumerate(data["follow_ups"], 1):
    print(f"{i}. {textwrap.fill(fu, width=76, subsequent_indent=' ' * (len(str(i)) + 2))}")
print("=" * 80)