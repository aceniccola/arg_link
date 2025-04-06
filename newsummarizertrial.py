# TODO: make this better!
import json
from models import high_thinking_model # Assuming this is correctly configured Gemini model
# from tools import tools # Assuming these are defined LangChain tools
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
    what: str = Field(description="A concise summary of the argument's core message. What is it really saying?", max_length=150) # Increased length slightly
    who: str = Field(description="Identify the speaker (or author) and their likely intended audience.", max_length=100)
    why: str = Field(description="Infer the likely motive or goal behind presenting this argument.", max_length=100)
    what_to_do: str = Field(description="Suggest the single most effective way to respond to or counter this argument.", max_length=150) # Increased length slightly

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

Structure your output as a JSON object matching the requested fields (what, who, why, what_to_do).
"""
summarizer_prompt = PromptTemplate.from_template(summarizer_prompt_template)

# --- Configure the model for structured output ---
# Ensure high_thinking_model is a compatible ChatModel instance
try:
    # Pass the Pydantic class to with_structured_output
    structured_llm = high_thinking_model.with_structured_output(Summary)
    print("Model configured for structured output with Summary class.")
except AttributeError:
    print("Error: 'high_thinking_model' does not seem to have 'with_structured_output'.")
    print("Ensure 'high_thinking_model' is a compatible LangChain ChatModel instance (e.g., ChatGoogleGenerativeAI).")
    # Handle the error appropriately, maybe exit or use a default model
    structured_llm = high_thinking_model # Fallback, likely won't work as intended
except Exception as e:
    print(f"An unexpected error occurred during model configuration: {e}")
    structured_llm = high_thinking_model # Fallback


# --- Create the agent with the structured LLM ---
# Pass the model configured for structured output
summarizer_graph = create_react_agent(
    model=structured_llm,
    tools=None,
    checkpointer=MemorySaver()
)
print("React agent created with structured LLM.")

config = {"configurable": {"thread_id": "thread-structured-1"}} # Use a different thread_id if needed

# (Keep print_stream function as is from the previous version if you still need it for debugging)
# def print_stream(graph, inputs, config): ...

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
    agent_input = {"messages": [HumanMessage(content=input_text)]}
    print(f"Invoking agent with input: {agent_input}") # Debug print
    try:
        # The result should ideally be the structured object (Summary instance)
        # or a dictionary that Pydantic can validate
        result = summarizer_graph.invoke(agent_input, config=config)

        # The actual output might be nested within the agent's final state
        # Check common patterns for the final answer in ReAct agents
        final_answer = None
        if isinstance(result, dict):
            if 'agent_outcome' in result and hasattr(result['agent_outcome'], 'return_values'):
                 # LangChain standard AgentExecutor output
                 output_key = next(iter(result['agent_outcome'].return_values), None)
                 if output_key:
                     final_answer = result['agent_outcome'].return_values[output_key]
            elif 'messages' in result and result['messages']:
                 # Check the content of the last AI message
                 last_message = result['messages'][-1]
                 if hasattr(last_message, 'content'):
                      # Sometimes the structured output is directly in the content
                      # Needs parsing if it's a stringified JSON
                      if isinstance(last_message.content, (dict, Summary)):
                           final_answer = last_message.content
                      elif isinstance(last_message.content, str):
                           try:
                               # Attempt to parse if it looks like JSON
                               potential_json = json.loads(last_message.content)
                               final_answer = potential_json
                           except json.JSONDecodeError:
                               print("Warning: Last message content is a string but not valid JSON.")
                               final_answer = result # Fallback to raw result
                      else:
                           final_answer = result # Fallback
                 else:
                     final_answer = result # Fallback
            else:
                 # If the result itself is the Summary object or a dict matching it
                 if isinstance(result, (Summary, dict)):
                      final_answer = result
                 else:
                     final_answer = result # Fallback

        elif isinstance(result, Summary):
             final_answer = result # Direct output is the Summary object
        else:
             final_answer = result # Fallback

        print("Agent invocation successful.")
        return final_answer

    except Exception as e:
        print(f"Error during agent invocation: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return None # Indicate failure

# (Keep print_summarizer if needed, update it similarly to handle structured output)

if __name__ == "__main__":
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
                                # If it's a dictionary, try accessing keys
                                print(f"  What: {summary_result.get('what', 'N/A')}")
                                print(f"  Who: {summary_result.get('who', 'N/A')}")
                                print(f"  Why: {summary_result.get('why', 'N/A')}")
                                print(f"  What To Do: {summary_result.get('what_to_do', 'N/A')}")
                                # Optionally, validate with Pydantic:
                                # try:
                                #    validated_summary = Summary(**summary_result)
                                #    print("(Dictionary validated successfully)")
                                # except Exception as validation_error:
                                #    print(f"(Dictionary validation failed: {validation_error})")
                            else:
                                print(f"  Received unexpected format: {type(summary_result)}")
                                print(f"  Raw Output: {summary_result}")
                        else:
                            print("  Summarizer did not return a result.")

                    except Exception as e:
                        print(f"Error processing moving brief argument {j+1}: {e}")
                        traceback.print_exc()
                else:
                    print(f"Skipping moving brief argument {j+1} due to missing or empty 'content'.")

        # Process Response Brief (similar logic)
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
                             else:
                                 print(f"  Received unexpected format: {type(summary_result)}")
                                 print(f"  Raw Output: {summary_result}")
                         else:
                             print("  Summarizer did not return a result.")

                     except Exception as e:
                         print(f"Error processing response brief argument {k+1}: {e}")
                         traceback.print_exc()
                 else:
                     print(f"Skipping response brief argument {k+1} due to missing or empty 'content'.")


    print("\n===== Processing Complete =====")
