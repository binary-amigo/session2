import os
from groq import Groq, APIError

# (Re-include get_groq_client and CODING_ASSISTANT_SYSTEM_PROMPT if running standalone)
# def get_groq_client(): ...
# CODING_ASSISTANT_SYSTEM_PROMPT = "..."
DEFAULT_MODEL = "llama3-8b-8192"

def filter_messages_for_api(messages):
    """
    Filters messages to include only 'role' and 'content' keys,
    and 'tool_calls' or 'tool_call_id' if present.
    This is useful if you add custom keys to your message objects locally.
    """
    api_messages = []
    for msg in messages:
        api_msg = {"role": msg["role"], "content": msg.get("content")}
        if "tool_calls" in msg and msg["tool_calls"] is not None:
            api_msg["tool_calls"] = msg["tool_calls"]
        if "tool_call_id" in msg and msg["tool_call_id"] is not None:
            # This is for messages with role 'tool'
            api_msg["tool_call_id"] = msg["tool_call_id"]
            # Content for role 'tool' is the result of the tool call
            api_msg["content"] = msg["content"]
        elif msg.get("content") is None and "tool_calls" not in msg:
            # If content is None and it's not a tool_calls request,
            # it might be an assistant message placeholder before content is filled.
            # For some APIs, sending `content: null` is fine, for others it might error.
            # Groq API expects content to be a string or null for assistant messages
            # that are expecting a tool_call.
            # If it's an assistant message that will have tool_calls, content can be null.
            if msg["role"] == "assistant":
                 api_msg["content"] = None # Explicitly set to None if no content and no tool_calls
            else:
                # For user messages, content should generally not be None.
                # For system messages, content is required.
                # This case might need specific handling based on API requirements.
                # For simplicity here, we'll ensure content is at least an empty string if None and not a tool_call.
                # However, the Groq API might require content to be non-null for user/system.
                # The original script's filter was simpler; this is more robust.
                # Let's stick to the original script's simplicity for this example:
                # if msg.get("content") is not None:
                # api_msg["content"] = msg["content"]
                # else:
                # api_msg["content"] = "" # Or handle as an error
                pass # Let Groq handle it if content is None and not a tool_call

        # Ensure content is not None for roles that require it, unless it's an assistant expecting a tool_call
        if msg["role"] != "assistant" and api_msg.get("content") is None and "tool_calls" not in api_msg:
            api_msg["content"] = "" # Default to empty string if None, adjust as needed

        api_messages.append(api_msg)
    return api_messages


def chat_with_history(client: Groq, conversation_history: list, new_user_query: str, model: str = DEFAULT_MODEL):
    """
    Conducts a chat turn, appending the new user query and assistant's response
    to the conversation history.
    """
    if not client:
        print("Groq client is not initialized.")
        return None, conversation_history

    # Add the new user query to the history
    conversation_history.append({"role": "user", "content": new_user_query})

    # Prepare messages for the API (system prompt is part of the history if added initially)
    # For this function, assume system prompt is the first message in conversation_history
    api_ready_messages = filter_messages_for_api(conversation_history)

    try:
        chat_completion = client.chat.completions.create(
            messages=api_ready_messages,
            model=model,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False,
        )
        assistant_response = chat_completion.choices[0].message.content
        # Add assistant's response to the history
        conversation_history.append({"role": "assistant", "content": assistant_response})
        return assistant_response, conversation_history
    except APIError as e:
        print(f"Groq API Error: {e}")
    except RateLimitError as e:
        print(f"Groq Rate Limit Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None, conversation_history

if __name__ == "__main__":
    # Example usage:
    # from your_module_1 import get_groq_client # Assuming get_groq_client is saved
    # client = get_groq_client()

    # For standalone testing:
    def get_groq_client_local():
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key: return None
        return Groq(api_key=api_key)
    client = get_groq_client_local()

    CODING_ASSISTANT_SYSTEM_PROMPT = """
    You are a specialized Coding Assistant AI. Your primary goal is to assist users with their coding-related questions.
    You must strictly adhere to the following guidelines:
    1.  **Scope of Assistance:** Only answer questions directly related to programming, software development, algorithms, data structures, coding tools (IDEs, compilers, debuggers, version control), APIs, SDKs, and software architecture.
    2.  **Refusal for Off-Topic Questions:** If a user asks a question outside this scope (e.g., about history, biology, general knowledge, opinions, personal advice), you MUST politely refuse to answer. You can say something like: "I am a specialized coding assistant and cannot answer questions outside of programming topics." or "My apologies, but I'm programmed to assist with coding-related queries only." Do NOT attempt to answer off-topic questions.
    """ # Abridged for brevity in example

    if client:
        # Initialize conversation history with the system prompt
        chat_log = [{"role": "system", "content": CODING_ASSISTANT_SYSTEM_PROMPT}]

        print("Starting chat session (type 'quit' to exit):")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break

            response, chat_log = chat_with_history(client, chat_log, user_input)

            if response:
                print(f"Assistant: {response}")
            else:
                print("Assistant: Sorry, I encountered an error.")
                # Optionally remove the last user message if the call failed
                if chat_log and chat_log[-1]["role"] == "user":
                    chat_log.pop()
        print("Chat session ended.")
    else:
        print("Failed to initialize Groq client. Cannot run example.")

