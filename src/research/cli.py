import anthropic

question = input("Enter your question: ")

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": question
        }
    ],
)

for block in response.content:
    if block.type == "text":
        print(block.text)

