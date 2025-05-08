import os
from groq import Groq, APIError, RateLimitError

# It's good practice to define constants for model names
# and other configurations.
DEFAULT_MODEL = "llama3-8b-8192"

def get_groq_client():
    """
    Initializes and returns a Groq API client.
    Assumes GROQ_API_KEY environment variable is set.
    """
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print(
                "GROQ_API_KEY not found in environment variables. "
                "Please set it."
            )
            return None
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        return None

if __name__ == "__main__":
    # Example usage:
    client = get_groq_client()
    if client:
        print("Successfully initialized Groq client!")
    else:
        print("Failed to initialize Groq client.")

