# TODO: make this better!
import json
# Make sure models and tools are correctly defined in their respective files
from models import high_thinking_model # Assuming this is correctly configured Gemini model (e.g., ChatGoogleGenerativeAI)
from tools import tools # Assuming these are defined LangChain tools - WE MAY NOT NEED THEM HERE
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
    # Handle the error appropriately, maybe exit or use a default model
    # For now, we'll let it proceed and likely fail at agent creation if structured_llm is None
    pass # Or exit()
except Exception as e:
    print(f"An unexpected error occurred during model configuration: {e}")
    # Handle appropriately
    pass # Or exit()


# --- Create the agent with the structured LLM ---
summarizer_graph = None
if structured_llm: # Proceed only if model configuration was successful
    try:
        # *** FIX: Pass an empty list for tools ***
        # Since the structured output mechanism is handled by the model binding,
        # we don't pass the external 'tools' list here unless the agent needs
        # *other* distinct tools for its intermediate steps. For pure structured
        # output based on the prompt, an empty list is usually correct.
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
    print(f"Invoking agent with input: {agent_input}") # Debug print
    try:
        # The result should ideally be the structured object (Summary instance)
        # or a dictionary that Pydantic can validate
        result = summarizer_graph.invoke(agent_input, config=config)

        # --- Refined output extraction ---
        # The output structure from create_react_agent often puts the final
        # response in the 'messages' list under the AI key.
        # If using .with_structured_output, the parsed object might be
        # directly in the content of the last AIMessage.

        final_answer = None
        if isinstance(result, dict) and "messages" in result:
            last_message = result["messages"][-1]
            # Check if the last message content is already the parsed Pydantic object
            if isinstance(last_message.content, Summary):
                 final_answer = last_message.content
            # Check if the content is a dictionary (might happen sometimes)
            elif isinstance(last_message.content, dict):
                 try:
                      final_answer = Summary(**last_message.content)
                      print("Parsed dictionary from last message content into Summary object.")
                 except Exception as pydantic_error:
                      print(f"Warning: Could not parse dict from last message into Summary: {pydantic_error}")
                      final_answer = last_message.content # Fallback to dict
            # Check if the content is a string (might be JSON)
            elif isinstance(last_message.content, str):
                 try:
                      # Attempt to parse if it looks like JSON
                      potential_json = json.loads(last_message.content)
                      final_answer = Summary(**potential_json)
                      print("Parsed JSON string from last message content into Summary object.")
                 except (json.JSONDecodeError, TypeError, ValueError) as parse_error:
                      print(f"Warning: Last message content is a string but not valid Summary JSON: {parse_error}")
                      # If it's not the structured output, maybe it's just a final text response
                      final_answer = last_message.content # Fallback to string
            else:
                 # Fallback for other unexpected content types
                 print(f"Warning: Unexpected content type in last message: {type(last_message.content)}")
                 final_answer = result # Fallback to the raw result dict

        elif isinstance(result, Summary):
             # Less common for create_react_agent, but possible
             final_answer = result
        else:
             # Fallback if the result structure is completely unexpected
             print(f"Warning: Unexpected result format from invoke: {type(result)}")
             final_answer = result # Fallback to raw result

        print("Agent invocation successful.")
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
        with open("stanford_hackathon_brief_pairs_clean.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: stanford_hackathon_brief_pairs_clean.json not found.")
        exit()
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from stanford_hackathon_brief_pairs_clean.json.")
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
                    print(f"\nArgument {j+1}:")
                    argument_text = content_item['content']
                    try:
                        prompt_value = summarizer_prompt.invoke({"argument": argument_text})
                        input_string = prompt_value.to_string()

                        # Call the summarizer expecting structured output
                        summary_result = call_summarizer(input_string, summarizer_graph=summarizer_graph, config=config)

                        # --- Handle the structured output ---
                        if summary_result:
                            print("Summary Received:")
                            if isinstance(summary_result, Summary):
                                # If it's already a Summary object
                                print(f"  What: {summary_result.what}")
                                print(f"  Who: {summary_result.who}")
                                print(f"  Why: {summary_result.why}")
                                print(f"  What To Do: {summary_result.what_to_do}")
                            elif isinstance(summary_result, dict):
                                # If it's a dictionary (less ideal fallback)
                                print(f"  What: {summary_result.get('what', 'N/A')}")
                                print(f"  Who: {summary_result.get('who', 'N/A')}")
                                print(f"  Why: {summary_result.get('why', 'N/A')}")
                                print(f"  What To Do: {summary_result.get('what_to_do', 'N/A')}")
                            elif isinstance(summary_result, str):
                                # If the fallback returned a string
                                print(f"  Received Text: {summary_result}")
                            else:
                                print(f"  Received unexpected format: {type(summary_result)}")
                                print(f"  Raw Output: {summary_result}")
                        else:
                            print("  Summarizer did not return a result or failed.")

                    except Exception as e:
                        print(f"Error processing moving brief argument {j+1}: {e}")
                        traceback.print_exc()
                else:
                    print(f"Skipping moving brief argument {j+1} due to missing or empty 'content'.")

        # Process Response Brief (similar logic - ensure robustness)
        if 'response_brief' in example and 'brief_arguments' in example['response_brief']:
            print("\n--- Response Brief Arguments ---")
            for k, content_item in enumerate(example['response_brief']['brief_arguments']):
                 if 'content' in content_item and content_item['content']:
                     print(f"\nArgument {k+1}:")
                     argument_text = content_item['content']
                     try:
                         prompt_value = summarizer_prompt.invoke({"argument": argument_text})
                         input_string = prompt_value.to_string()
                         summary_result = call_summarizer(input_string, summarizer_graph=summarizer_graph, config=config)

                         if summary_result:
                             print("Summary Received:")
                             if isinstance(summary_result, Summary):
                                 print(f"  What: {summary_result.what}")
                                 print(f"  Who: {summary_result.who}")
                                 print(f"  Why: {summary_result.why}")
                                 print(f"  What To Do: {summary_result.what_to_do}")
                             elif isinstance(summary_result, dict):
                                 print(f"  What: {summary_result.get('what', 'N/A')}")
                                 print(f"  Who: {summary_result.get('who', 'N/A')}")
                                 print(f"  Why: {summary_result.get('why', 'N/A')}")
                                 print(f"  What To Do: {summary_result.get('what_to_do', 'N/A')}")
                             elif isinstance(summary_result, str):
                                print(f"  Received Text: {summary_result}")
                             else:
                                 print(f"  Received unexpected format: {type(summary_result)}")
                                 print(f"  Raw Output: {summary_result}")
                         else:
                             print("  Summarizer did not return a result or failed.")

                     except Exception as e:
                         print(f"Error processing response brief argument {k+1}: {e}")
                         traceback.print_exc()
                 else:
                     print(f"Skipping response brief argument {k+1} due to missing or empty 'content'.")


    print("\n===== Processing Complete =====")
