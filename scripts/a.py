# TODO: make this better!
import json
# Make sure models and tools are correctly defined in their respective files
from models import high_thinking_model # Assuming this is correctly configured Gemini model (e.g., ChatGoogleGenerativeAI)
# *** FIX: Remove unused import if tools are not needed for this specific agent ***
# from tools import tools
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
# Import Pydantic classes for structured output
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import traceback # For more detailed error printing

# --- Define the desired output structure ---
class Summary(BaseModel):
    """Summary to assist in decision making regarding a specific argument.""" # Docstring helps the model
    what: str = Field(description="A concise summary of the argument's core message. What is it really saying?", max_length=150)
    who: str = Field(description="Identify the speaker (or author) and their likely intended audience.", max_length=100)
    why: str = Field(description="Infer the likely motive or goal behind presenting this argument.", max_length=100)
    what_to_do: str = Field(description="Suggest the single most effective way to respond to or counter this argument.", max_length=150)

# --- Updated Prompt Template ---
# Explicitly mention the desired output structure and fields
summarizer_prompt_template = """
Your task is to analyze and summarize the provided argument. Structure your response precisely according to the following fields:
- what: What is this argument really saying? Summarize its core point concisely.
- who: Who is the speaker (or author) and who are they likely trying to reach (their audience)?
- why: What is the likely motive or underlying goal behind making this argument?
- what_to_do: What is the single most effective way to respond to or counter this specific argument?

Provide your analysis based *only* on the following argument:
Argument: {argument}

Structure your output as a JSON object matching the requested fields (what, who, why, what_to_do). Do not use any other external tools.
""" # Added instruction to not use other tools if only summarization is needed
summarizer_prompt = PromptTemplate.from_template(summarizer_prompt_template)

# --- Configure the model for structured output ---
# Ensure high_thinking_model is a compatible ChatModel instance
structured_llm = None
try:
    # Pass the Pydantic class to with_structured_output
    # This implicitly binds a tool for the 'Summary' structure to the LLM
    structured_llm = high_thinking_model.with_structured_output(Summary)
    print("Model configured for structured output with Summary class.")
except AttributeError:
    print("Error: 'high_thinking_model' does not seem to have 'with_structured_output'.")
    print("Ensure 'high_thinking_model' is a compatible LangChain ChatModel instance (e.g., ChatGoogleGenerativeAI).")
    pass # Or exit()
except Exception as e:
    print(f"An unexpected error occurred during model configuration: {e}")
    pass # Or exit()


# --- Create the agent with the structured LLM ---
summarizer_graph = None
if structured_llm: # Proceed only if model configuration was successful
    try:
        # Pass an empty list for tools since the prompt forbids external tool use
        # and the structure is handled by the LLM binding.
        summarizer_graph = create_react_agent(
            model=structured_llm,
            tools=[], # Pass empty list here
            checkpointer=MemorySaver()
        )
        print("React agent created with structured LLM and no additional external tools.")
    except Exception as e:
        print(f"Error creating react agent: {e}")
        traceback.print_exc()
else:
    print("Agent creation skipped due to model configuration error.")


config = {"configurable": {"thread_id": "thread-structured-1"}}

# --- Modified call_summarizer to handle potential structured output ---
def call_summarizer(input_text: str, summarizer_graph = summarizer_graph, config = config):
    """
    Invokes the summarizer graph with the input text formatted correctly,
    expecting a structured output.

    Args:
        input_text: The string content to be summarized.
        summarizer_graph: The LangGraph agent executor configured for structured output.
        config: The configuration dictionary for the graph invocation.

    Returns:
        The result from graph.invoke(), hopefully an instance of Summary or a dict.
    """
    # Ensure the graph was created successfully
    if not summarizer_graph:
        print("Error: Summarizer graph is not initialized.")
        return None

    agent_input = {"messages": [HumanMessage(content=input_text)]}
    # print(f"Invoking agent with input: {agent_input}") # Keep this commented unless debugging verbosely
    try:
        result = summarizer_graph.invoke(agent_input, config=config)

        # --- Refined output extraction ---
        final_answer = None
        if isinstance(result, dict) and "messages" in result:
            last_message = result["messages"][-1]
            if isinstance(last_message.content, Summary):
                 final_answer = last_message.content
            elif isinstance(last_message.content, dict):
                 try:
                      final_answer = Summary(**last_message.content)
                      # print("Parsed dictionary from last message content into Summary object.") # Concise
                 except Exception as pydantic_error:
                      print(f"Warning: Could not parse dict from last message into Summary: {pydantic_error}")
                      final_answer = last_message.content # Fallback to dict
            elif isinstance(last_message.content, str):
                 try:
                      potential_json = json.loads(last_message.content)
                      final_answer = Summary(**potential_json)
                      # print("Parsed JSON string from last message content into Summary object.") # Concise
                 except (json.JSONDecodeError, TypeError, ValueError) as parse_error:
                      print(f"Warning: Last message content is a string but not valid Summary JSON: {parse_error}")
                      final_answer = last_message.content # Fallback to string
            else:
                 print(f"Warning: Unexpected content type in last message: {type(last_message.content)}")
                 final_answer = result # Fallback to the raw result dict
        elif isinstance(result, Summary):
             final_answer = result
        else:
             print(f"Warning: Unexpected result format from invoke: {type(result)}")
             final_answer = result # Fallback to raw result

        # print("Agent invocation successful.") # Concise
        return final_answer

    except Exception as e:
        print(f"Error during agent invocation: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return None # Indicate failure

# (Keep print_summarizer if needed, update it similarly to handle structured output)

if __name__ == "__main__":
    # Ensure agent is initialized before proceeding
    if not summarizer_graph:
        print("Exiting: Summarizer agent could not be initialized.")
        exit()

    try:
        # Use a relative path or ensure the absolute path is correct
        file_path = "stanford_hackathon_brief_pairs_clean.json"
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred loading data: {e}")
        exit()

    examples_to_process = 1 # Adjust as needed

    for i, example in enumerate(data):
        if i >= examples_to_process:
            break

        print(f"\n===== Processing Example {i+1} =====")

        # Process Moving Brief
        if 'moving_brief' in example and 'brief_arguments' in example['moving_brief']:
            print("\n--- Moving Brief Arguments ---")
            for j, content_item in enumerate(example['moving_brief']['brief_arguments']):
                if 'content' in content_item and content_item['content']: # Check if content exists and is not empty
                    # print(f"\nArgument {j+1}:") # Concise
                    argument_text = content_item['content']
                    try:
                        prompt_value = summarizer_prompt.invoke({"argument": argument_text})
                        input_string = prompt_value.to_string()
                        summary_result = call_summarizer(input_string, summarizer_graph=summarizer_graph, config=config)

                        # --- Handle the structured output ---
                        if summary_result:
                            # print("Summary Received:") # Concise
                            if isinstance(summary_result, Summary):
                                print(f"  Summary (Arg {j+1}): What: {summary_result.what[:50]}... Who: {summary_result.who[:50]}... Why: {summary_result.why[:50]}... ToDo: {summary_result.what_to_do[:50]}...") # Print concisely
                            elif isinstance(summary_result, dict):
                                print(f"  Dict Summary (Arg {j+1}): {str(summary_result)[:100]}...") # Concise
                            elif isinstance(summary_result, str):
                                print(f"  Text Summary (Arg {j+1}): {summary_result[:100]}...") # Concise
                            else:
                                print(f"  Unexpected Output (Arg {j+1}): {type(summary_result)} {str(summary_result)[:100]}...")
                        else:
                            print(f"  Summarizer failed for Arg {j+1}.")

                    except Exception as e:
                        print(f"Error processing moving brief argument {j+1}: {e}")
                        # traceback.print_exc() # Comment out for less verbose errors
                else:
                    # print(f"Skipping moving brief argument {j+1} due to missing or empty 'content'.") # Concise
                    pass

        # Process Response Brief (similar logic - ensure robustness)
        if 'response_brief' in example and 'brief_arguments' in example['response_brief']:
            print("\n--- Response Brief Arguments ---")
            for k, content_item in enumerate(example['response_brief']['brief_arguments']):
                 if 'content' in content_item and content_item['content']:
                     # print(f"\nArgument {k+1}:") # Concise
                     argument_text = content_item['content']
                     try:
                         prompt_value = summarizer_prompt.invoke({"argument": argument_text})
                         input_string = prompt_value.to_string()
                         summary_result = call_summarizer(input_string, summarizer_graph=summarizer_graph, config=config)

                         if summary_result:
                             # print("Summary Received:") # Concise
                             if isinstance(summary_result, Summary):
                                 print(f"  Summary (Arg {k+1}): What: {summary_result.what[:50]}... Who: {summary_result.who[:50]}... Why: {summary_result.why[:50]}... ToDo: {summary_result.what_to_do[:50]}...") # Print concisely
                             elif isinstance(summary_result, dict):
                                 print(f"  Dict Summary (Arg {k+1}): {str(summary_result)[:100]}...") # Concise
                             elif isinstance(summary_result, str):
                                 print(f"  Text Summary (Arg {k+1}): {summary_result[:100]}...") # Concise
                             else:
                                 print(f"  Unexpected Output (Arg {k+1}): {type(summary_result)} {str(summary_result)[:100]}...")
                         else:
                             print(f"  Summarizer failed for Arg {k+1}.")

                     except Exception as e:
                         print(f"Error processing response brief argument {k+1}: {e}")
                         # traceback.print_exc() # Comment out for less verbose errors
                 else:
                     # print(f"Skipping response brief argument {k+1} due to missing or empty 'content'.") # Concise
                     pass


    print("\n===== Processing Complete =====")
