import json
import traceback
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Import your specific model and tools
from models import high_thinking_model # Ensure this is a compatible ChatModel (e.g., ChatGoogleGenerativeAI)
from tools import tools # Import the functional tools list

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# --- Define the desired output structure ---
class Summary(BaseModel):
    """Summary to assist in decision making regarding a specific argument."""
    what: str = Field(description="A concise summary of the argument's core message. What is it really saying?", max_length=150)
    who: str = Field(description="Identify the speaker (or author) and their likely intended audience.", max_length=100)
    why: str = Field(description="Infer the likely motive or goal behind presenting this argument.", max_length=100)
    what_to_do: str = Field(description="Suggest the single most effective way to respond to or counter this argument.", max_length=150)

tools.append(Summary)

# --- Updated Prompt Template ---
# Allows tool use, still guides towards final Summary structure
summarizer_prompt_template = """
Your task is to analyze and summarize the provided argument, using available tools if necessary to gather context or clarify information. Structure your final response precisely according to the following fields:
- what: What is this argument really saying? Summarize its core point concisely.
- who: Who is the speaker (or author) and who are they likely trying to reach (their audience)?
- why: What is the likely motive or underlying goal behind making this argument?
- what_to_do: What is the single most effective way to respond to or counter this specific argument?

Provide your analysis based on the following argument:
Argument: {argument}

Structure your final output as a JSON object matching the requested fields (what, who, why, what_to_do).
"""
summarizer_prompt = PromptTemplate.from_template(summarizer_prompt_template)

# --- Configure the model for structured output AND bind tools ---
structured_llm_with_tools = None
try:
    # Bind both the Pydantic class (for output structure) and the functional tools
    # Note: Ensure 'tools' contains valid LangChain tools
    all_bound_items = tools
    structured_llm_with_tools = high_thinking_model.bind_tools(all_bound_items)
    # print("Model configured with bound tools and structured output.") # Concise
except AttributeError:
    print("Error: 'high_thinking_model' does not seem to have 'bind_tools'.")
    print("Ensure 'high_thinking_model' is a compatible LangChain ChatModel instance.")
except Exception as e:
    print(f"An unexpected error occurred during model configuration: {e}")

# --- Create the agent with the configured LLM and functional tools ---
summarizer_graph = None
if structured_llm_with_tools:
    try:
        # Pass the LLM with tools+structure bound to it.
        # Pass ONLY the functional 'tools' list for the agent to execute.
        summarizer_graph = create_react_agent(
            model=structured_llm_with_tools,
            tools=tools, # Pass the executable tools here
            checkpointer=MemorySaver()
        )
        # print("React agent created with bound LLM and functional tools.") # Concise
    except Exception as e:
        print(f"Error creating react agent: {e}")
        traceback.print_exc()
else:
    print("Agent creation skipped due to model configuration error.")

config = {"configurable": {"thread_id": "thread-structured-tools-1"}} # New thread ID

# --- call_summarizer function (less verbose) ---
def call_summarizer(input_text: str, summarizer_graph = summarizer_graph, config = config):
    """Invokes the summarizer graph, expecting a structured output."""
    if not summarizer_graph:
        print("Error: Summarizer graph is not initialized.")
        return None

    agent_input = {"messages": [HumanMessage(content=input_text)]}
    try:
        result = summarizer_graph.invoke(agent_input, config=config)
    
        # Extract final answer (might be Summary object, dict, or string)
        final_answer = None
        if isinstance(result, dict) and "messages" in result:
            last_message = result["messages"][-1]
            # Check common places for the final structured output or text
            if isinstance(last_message.content, Summary):
                 final_answer = last_message.content
            elif isinstance(last_message.content, dict):
                 try: # Attempt to cast dict to Summary
                      final_answer = Summary(**last_message.content)
                 except Exception:
                      final_answer = last_message.content # Fallback to dict
            elif isinstance(last_message.content, str):
                 try: # Attempt to parse string as JSON -> Summary
                      potential_json = json.loads(last_message.content)
                      final_answer = Summary(**potential_json)
                 except Exception:
                      final_answer = last_message.content # Fallback to string
            else: # Fallback for other types
                 final_answer = result
        elif isinstance(result, Summary): # Direct Summary object output
             final_answer = result
        else: # Fallback for completely unexpected result format
             final_answer = result
        print("------------ result size: ",len(result))
        return final_answer

    except Exception as e:
        print(f"Error during agent invocation: {e}")
        # traceback.print_exc() # Keep commented for concise output
        return None

# --- Main execution block (less verbose) ---
if __name__ == "__main__":
    if not summarizer_graph:
        print("Exiting: Summarizer agent could not be initialized.")
        exit()

    try:
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

    examples_to_process = 10 # Adjust as needed
    with open("stanford_hackathon_brief_pairs.json", "r") as r:
        data = json.load(r)
    with open("output.json", "w") as w:
        w.write(json.dumps(data))
    for i, example in enumerate(data):
        if i >= examples_to_process:
            break
        print(f"\n===== Processing Example {i+1} =====")

        # Process Moving Brief
        if 'moving_brief' in example and 'brief_arguments' in example['moving_brief']:
            print("\n--- Moving Brief Arguments ---")
            for j, content_item in enumerate(example['moving_brief']['brief_arguments']):
                if 'content' in content_item and content_item['content']:
                    argument_text = content_item['content']
                    try:
                        prompt_value = summarizer_prompt.invoke({"argument": argument_text})
                        input_string = prompt_value.to_string()
                        summary_result = call_summarizer(input_string, summarizer_graph=summarizer_graph, config=config)
                        if summary_result:
                            if isinstance(summary_result, Summary): 
                                print(f"  Summary (Arg {j+1}):")
                                print(f"    What: {summary_result.what}")
                                print(f"    Who: {summary_result.who}")
                                print(f"    Why: {summary_result.why}")
                                print(f"    ToDo: {summary_result.what_to_do}")
                            elif isinstance(summary_result, (dict, str)):
                                print(f"  Fallback Output (Arg {j+1}): {summary_result}")
                            else:
                                print(f"  Unexpected Output (Arg {j+1}): {type(summary_result)} {summary_result}")
                        else:
                            print(f"  Summarizer failed for Arg {j+1}.")
                    except Exception as e:
                        print(f"Error processing moving brief argument {j+1}: {e}")
                else:
                    pass # Skip silently

        # Process Response Brief
        if 'response_brief' in example and 'brief_arguments' in example['response_brief']:
            print("\n--- Response Brief Arguments ---")
            for k, content_item in enumerate(example['response_brief']['brief_arguments']):
                 if 'content' in content_item and content_item['content']:
                     argument_text = content_item['content']
                     try:
                         prompt_value = summarizer_prompt.invoke({"argument": argument_text})
                         input_string = prompt_value.to_string()
                         summary_result = call_summarizer(input_string, summarizer_graph=summarizer_graph, config=config)

                         if summary_result:
                             if isinstance(summary_result, Summary):
                                 print(f"  Summary (Arg {k+1}):")
                                 print(f"    What: {summary_result.what}")
                                 print(f"    Who: {summary_result.who}")
                                 print(f"    Why: {summary_result.why}")
                                 print(f"    ToDo: {summary_result.what_to_do}")
                             elif isinstance(summary_result, (dict, str)):
                                 print(f"  Fallback Output (Arg {k+1}): {summary_result}")
                             else:
                                 print(f"  Unexpected Output (Arg {k+1}): {type(summary_result)} {summary_result}")
                         else:
                             print(f"  Summarizer failed for Arg {k+1}.")
                     except Exception as e:
                         print(f"Error processing response brief argument {k+1}: {e}")
                 else:
                     pass # Skip silently

    print("\n===== Processing Complete =====")
