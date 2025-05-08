import os
from groq import Groq, APIError

# (Re-include get_groq_client from Module 1 if running standalone)
# def get_groq_client(): ...

DEFAULT_MODEL = "llama3-8b-8192" # Or any other model you prefer

# System prompt defining the AI's role and behavior
CODING_ASSISTANT_SYSTEM_PROMPT = """
You are a specialized Coding Assistant AI. Your primary goal is to assist users with their coding-related questions.
You must strictly adhere to the following guidelines:
1.  **Scope of Assistance:** Only answer questions directly related to programming, software development, algorithms, data structures, coding tools (IDEs, compilers, debuggers, version control), APIs, SDKs, and software architecture.
2.  **Refusal for Off-Topic Questions:** If a user asks a question outside this scope (e.g., about history, biology, general knowledge, opinions, personal advice), you MUST politely refuse to answer. You can say something like: "I am a specialized coding assistant and cannot answer questions outside of programming topics." or "My apologies, but I'm programmed to assist with coding-related queries only." Do NOT attempt to answer off-topic questions.
3.  **Accuracy and Clarity:** Provide accurate, clear, and concise explanations. If you provide code snippets, ensure they are correct and well-explained.
4.  **No Personal Opinions:** Do not express personal opinions or engage in speculative discussions.
5.  **Professional Tone:** Maintain a professional and helpful tone at all times.
"""

def ask_llm_basic(client: Groq, user_query: str, model: str = DEFAULT_MODEL):
    """
    Sends a single user query to the LLM with a system prompt.
    """
    if not client:
        print("Groq client is not initialized.")
        return None

    messages = [
        {"role": "system", "content": CODING_ASSISTANT_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.7, # Controls randomness: lower is more deterministic
            max_tokens=1024, # Max length of the response
            top_p=1,         # Nucleus sampling
            stream=False,    # We'll handle non-streaming for simplicity here
        )
        return chat_completion.choices[0].message.content
    except APIError as e:
        print(f"Groq API Error: {e}")
    except RateLimitError as e:
        print(f"Groq Rate Limit Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

if __name__ == "__main__":
    # Example usage:
    # Ensure GROQ_API_KEY is set in your environment
    # from your_module_1 import get_groq_client # Assuming get_groq_client is saved
    # client = get_groq_client()

    # For standalone testing, you can define get_groq_client here:
    def get_groq_client_local():
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key: return None
        return Groq(api_key=api_key)
    client = get_groq_client_local()

    if client:
        question = "Explain what a list comprehension is in Python."
        # question = "What is the capital of France?" # To test refusal
        response = ask_llm_basic(client, question)
        if response:
            print(f"User: {question}")
            print(f"Assistant: {response}")
    else:
        print("Failed to initialize Groq client. Cannot run example.")
