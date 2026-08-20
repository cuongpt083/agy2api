import os
from openai import OpenAI

# Initialize the OpenAI client pointing to the local AGY2API wrapper
client = OpenAI(
    base_url="http://localhost:8000/v1",
    # Pass your configured AGY_API_KEY here
    api_key=os.environ.get("AGY_API_KEY", "your-secret-key-here")
)

def main():
    print("Sending streaming chat completion request to AGY2API...")
    
    stream = client.chat.completions.create(
        model="Gemini 3.6 Flash (High)",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short haiku about coding."}
        ],
        stream=True,
    )

    print("\nResponse:")
    print("-" * 20)
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end="", flush=True)
    print()
    print("-" * 20)

if __name__ == "__main__":
    main()
