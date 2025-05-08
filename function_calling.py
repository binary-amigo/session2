import os
import json
from datetime import datetime
from groq import Groq, APIError

# (Re-include get_groq_client, CODING_ASSISTANT_SYSTEM_PROMPT, filter_messages_for_api if running standalone)
DEFAULT_MODEL = "llama3-8b-8192"

# 1. Define the function our AI can call
def get_current_datetime():
    """Returns the current date and time as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 2. Define the schema for the tools the LLM can use
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time. Use this for any questions related to the current time.",
            "parameters": {
                "type": "object",
                "properties": {}, # No parameters for this simple function
                "required": [],
            },
        },
    }
]

# Available functions mapping
AVAILABLE_FUNCTIONS = {
    "get_current_datetime": get_current_datetime,
}

def run_conversation_with_tools(client: Groq, conversation_history: list, new_user_query: str, model: str = DEFAULT_MODEL):
    """
    Handles a conversation turn, including potential function calls.
    """
    if not client:
        print("Groq client not initialized.")
        return None, conversation_history

    conversation_history.append({"role": "user", "content": new_user_query})
    
    # Ensure system prompt is the first message
    # (Assuming it's already there from initialization)
    api_ready_messages = filter_messages_for_api(conversation_history)

    try:
        # First API call: Allow the LLM to decide if it needs to use a tool
        print("--- Sending to LLM (tool_choice='auto') ---")
        # print(f"Messages sent: {json.dumps(api_ready_messages, indent=2)}")

        chat_completion = client.chat.completions.create(
            messages=api_ready_messages,
            model=model,
            tools=TOOLS_SCHEMA,
            tool_choice="auto", # "auto" lets the model decide, "none" forces no tools
            temperature=0.7,
            max_tokens=1024,
        )
        response_message = chat_completion.choices[0].message
        # print(f"LLM raw response: {response_message}")


        # 3. Check if the LLM wants to call a tool
        if response_message.tool_calls:
            print("--- LLM requested a tool call ---")
            # Append the assistant's intent to call a function (contains tool_calls)
            conversation_history.append(
                {
                    "role": "assistant",
                    "content": response_message.content, # May be None
                    "tool_calls": response_message.tool_calls,
                }
            )

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                # function_args = json.loads(tool_call.function.arguments) # If args were expected

                if function_name in AVAILABLE_FUNCTIONS:
                    # 4. Execute the function
                    function_to_call = AVAILABLE_FUNCTIONS[function_name]
                    print(f"Executing function: {function_name}")
                    try:
                        # For simplicity, assuming no arguments for get_current_datetime
                        function_response = function_to_call()
                        print(f"Function response: {function_response}")

                        # 5. Send the function's output back to the LLM
                        conversation_history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": function_response,
                            }
                        )
                    except Exception as e:
                        print(f"Error executing function {function_name}: {e}")
                        conversation_history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": f"Error: {e}",
                            }
                        )
                else:
                    print(f"Unknown function requested: {function_name}")
                    conversation_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": f"Error: Unknown function '{function_name}'",
                        }
                    )
            
            # Second API call: LLM processes the tool's response
            print("--- Sending tool response to LLM (tool_choice='none') ---")
            api_ready_messages_after_tool = filter_messages_for_api(conversation_history)
            # print(f"Messages sent: {json.dumps(api_ready_messages_after_tool, indent=2)}")

            final_completion = client.chat.completions.create(
                messages=api_ready_messages_after_tool,
                model=model,
                tool_choice="none", # Important: prevent recursion
                temperature=0.7,
                max_tokens=1024,
            )
            final_response_content = final_completion.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": final_response_content})
            return final_response_content, conversation_history
        else:
            # No tool call, just a direct answer
            print("--- LLM provided a direct answer ---")
            assistant_response_content = response_message.content
            conversation_history.append({"role": "assistant", "content": assistant_response_content})
            return assistant_response_content, conversation_history

    except APIError as e:
        print(f"Groq API Error: {e}")
    except RateLimitError as e:
        print(f"Groq Rate Limit Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None, conversation_history


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

    CODING_ASSISTANT_SYSTEM_PROMPT = """
    You are a specialized Coding Assistant AI. Your primary goal is to assist users with their coding-related questions.
    You must strictly adhere to the following guidelines:
    1.  **Scope of Assistance:** Only answer questions directly related to programming, software development, algorithms, data structures, coding tools (IDEs, compilers, debuggers, version control), APIs, SDKs, and software architecture.
    2.  **Refusal for Off-Topic Questions:** If a user asks a question outside this scope (e.g., about history, biology, general knowledge, opinions, personal advice), you MUST politely refuse to answer. You can say something like: "I am a specialized coding assistant and cannot answer questions outside of programming topics." or "My apologies, but I'm programmed to assist with coding-related queries only." Do NOT attempt to answer off-topic questions.
    3.  **Function Calling:** You have access to a tool called 'get_current_datetime'. Use it if the user's query implies needing the current time to answer a coding-related question (e.g., "Are there any Python conferences happening next week based on today's date?").
    """ # Updated for tool use

    # (Include filter_messages_for_api if not imported)
    def filter_messages_for_api(messages):
        api_messages = []
        for msg in messages:
            api_msg = {"role": msg["role"]}
            if "content" in msg and msg["content"] is not None:
                api_msg["content"] = msg["content"]
            if "tool_calls" in msg and msg["tool_calls"] is not None:
                api_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg["tool_calls"]
                ]
            if msg["role"] == "tool": # tool messages have name and tool_call_id
                api_msg["tool_call_id"] = msg["tool_call_id"]
                api_msg["name"] = msg["name"]
            # Ensure content is not None for roles that require it, unless it's an assistant expecting a tool_call
            if msg["role"] != "assistant" and api_msg.get("content") is None and "tool_calls" not in api_msg and msg["role"] != "tool":
                api_msg["content"] = "" 
            elif msg["role"] == "assistant" and "tool_calls" in msg and msg.get("content") is None:
                 api_msg["content"] = None # Groq allows this for assistant message with tool_calls
            api_messages.append(api_msg)
        return api_messages


    if client:
        chat_log = [{"role": "system", "content": CODING_ASSISTANT_SYSTEM_PROMPT}]
        
        # query = "What time is it right now?"
        query = "Can you tell me the current date and time to help me timestamp a log file in Python?"
        # query = "What is a Python dictionary?" # Test non-tool use

        print(f"User: {query}")
        response, chat_log = run_conversation_with_tools(client, chat_log, query)

        if response:
            print(f"Assistant: {response}")
        else:
            print("Assistant: Sorry, I encountered an error.")
        
        # print("\n--- Final Chat Log ---")
        # for message in chat_log:
        #     print(message)
    else:
        print("Failed to initialize Groq client. Cannot run example.")
