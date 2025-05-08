import os
import json
from groq import Groq, APIError

# (Re-include get_groq_client if running standalone)
DEFAULT_MODEL = "llama3-8b-8192" # Model for the main assistant
EVALUATION_MODEL = "llama3-8b-8192" # Can be the same or different

EVALUATION_SYSTEM_PROMPT = """
You are an Evaluation AI. Your task is to evaluate the Coding Assistant's response to a user's query.
The Coding Assistant is programmed to ONLY answer coding-related questions and politely refuse others.

Please evaluate based on the following criteria:
1.  **Coding Relevance (is_coding_related)**:
    *   `true`: If the USER'S QUERY is primarily about programming, software development, algorithms, data structures, coding tools, APIs, SDKs, or software architecture.
    *   `false`: If the USER'S QUERY is off-topic (e.g., history, general knowledge, personal advice).
2.  **Helpfulness (helpfulness_rating)**:
    *   If `is_coding_related` is `true`, rate the ASSISTANT'S RESPONSE from 1 (not helpful) to 5 (very helpful).
    *   If `is_coding_related` is `false`, this should be `null` or not applicable.
3.  **Refusal Appropriateness (refusal_appropriateness)**:
    *   If `is_coding_related` is `false`, did the ASSISTANT'S RESPONSE politely refuse to answer the off-topic query? (`true` or `false`).
    *   If `is_coding_related` is `true`, this should be `null` or not applicable.
4.  **Reasoning (reasoning)**: Briefly explain your ratings.

Output your evaluation in JSON format like this:
{
  "is_coding_related": boolean,
  "helpfulness_rating": integer | null,
  "refusal_appropriateness": boolean | null,
  "reasoning": "Your brief explanation here."
}
"""

def evaluate_response(client: Groq, user_query: str, assistant_response: str, model: str = EVALUATION_MODEL):
    """
    Uses an LLM to evaluate the assistant's response to a user query.
    """
    if not client:
        print("Groq client is not initialized for evaluation.")
        return None

    evaluation_prompt_messages = [
        {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"User Query: \"{user_query}\"\n\nAssistant Response: \"{assistant_response}\"\n\nPlease provide your evaluation in JSON format.",
        },
    ]

    try:
        # Ensure the model is instructed to output JSON
        # Some models support a response_format parameter.
        # For Groq with Llama3, explicitly asking in the prompt is key.
        chat_completion = client.chat.completions.create(
            messages=evaluation_prompt_messages,
            model=model,
            temperature=0.2, # Low temperature for more deterministic JSON
            max_tokens=500,
            # response_format={"type": "json_object"}, # If supported by model/Groq for this model
        )
        evaluation_content = chat_completion.choices[0].message.content

        # Attempt to parse the JSON from the response
        # LLMs might sometimes include explanations around the JSON block.
        try:
            # Find the start and end of the JSON block
            json_start = evaluation_content.find('{')
            json_end = evaluation_content.rfind('}') + 1
            if json_start != -1 and json_end != 0 and json_end > json_start:
                json_string = evaluation_content[json_start:json_end]
                parsed_evaluation = json.loads(json_string)
                return parsed_evaluation
            else:
                print("Could not find JSON block in evaluation response.")
                print(f"Raw evaluation response: {evaluation_content}")
                return {"error": "Failed to parse JSON, block not found", "raw_response": evaluation_content}

        except json.JSONDecodeError as e:
            print(f"JSONDecodeError in evaluation: {e}")
            print(f"Raw evaluation response: {evaluation_content}")
            return {"error": "Failed to parse JSON", "raw_response": evaluation_content}

    except APIError as e:
        print(f"Groq API Error during evaluation: {e}")
    except RateLimitError as e:
        print(f"Groq Rate Limit Error during evaluation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during evaluation: {e}")
    return None

if __name__ == "__main__":
    # Example usage:
    # from your_module_1 import get_groq_client
    # client = get_groq_client()

    # For standalone testing:
    def get_groq_client_local():
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key: return None
        return Groq(api_key=api_key)
    client = get_groq_client_local()

    if client:
        # Scenario 1: Coding question, good answer
        query1 = "What is a decorator in Python?"
        response1 = "A decorator in Python is a design pattern that allows you to modify or enhance functions or methods in a clean and readable way. It's a callable that takes another function as an argument (the decorated function) and returns a new function or modifies the original one."
        evaluation1 = evaluate_response(client, query1, response1)
        print(f"\n--- Evaluation for Query 1 ---")
        print(f"User Query: {query1}")
        print(f"Assistant Response: {response1}")
        print(f"Evaluation: {json.dumps(evaluation1, indent=2)}")

        # Scenario 2: Off-topic question, good refusal
        query2 = "What's the weather like today?"
        response2 = "My apologies, but I'm programmed to assist with coding-related queries only."
        evaluation2 = evaluate_response(client, query2, response2)
        print(f"\n--- Evaluation for Query 2 ---")
        print(f"User Query: {query2}")
        print(f"Assistant Response: {response2}")
        print(f"Evaluation: {json.dumps(evaluation2, indent=2)}")

        # Scenario 3: Coding question, poor/irrelevant answer
        query3 = "How do I sort a list in Python?"
        response3 = "Paris is the capital of France." # Clearly wrong for the query
        evaluation3 = evaluate_response(client, query3, response3)
        print(f"\n--- Evaluation for Query 3 ---")
        print(f"User Query: {query3}")
        print(f"Assistant Response: {response3}")
        print(f"Evaluation: {json.dumps(evaluation3, indent=2)}")

    else:
        print("Failed to initialize Groq client. Cannot run evaluation example.")

