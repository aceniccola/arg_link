import json
import traceback
import argparse
import os
import copy # Added for deepcopy
from typing import List, Dict, Any

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

# --- Updated Prompt Template ---
# Guides towards final Summary structure, allows tool use implicitly via ReAct agent
summarizer_prompt_template = """
Your task is to analyze and summarize the provided argument. Use available tools ONLY if necessary to gather external context or clarify specific information mentioned in the argument text itself. Focus primarily on the text provided.

Structure your final response precisely according to the following fields:
- what: What is this argument really saying? Summarize its core point concisely.
- who: Who is the speaker (or author) and who are they likely trying to reach (their audience)?
- why: What is the likely motive or underlying goal behind making this argument?
- what_to_do: What is the single most effective way to respond to or counter this specific argument?

Analyze the following argument:
Argument: {argument}

Generate a final answer containing ONLY the JSON object matching the requested fields (what, who, why, what_to_do). Do not include explanations before or after the JSON object in the final answer. If you use tools, provide reasoning during intermediate steps, but the final output must be just the JSON.
"""
summarizer_prompt = PromptTemplate.from_template(summarizer_prompt_template)

# --- Configure the model for structured output ---
# We will use the prompt to guide the ReAct agent towards the JSON output.
# Binding structured output directly to a ReAct agent can be complex.
# The prompt now explicitly asks for the JSON in the final answer.
configured_model = high_thinking_model # Use the base model

# --- Create the agent ---
summarizer_graph = None
try:
    # Pass the base model and the functional tools
    summarizer_graph = create_react_agent(
        model=configured_model,
        tools=tools, # Pass the executable tools here
        checkpointer=MemorySaver()
    )
except Exception as e:
    print(f"Error creating react agent: {e}")
    traceback.print_exc()

config = {"configurable": {"thread_id": "thread-summarizer-file-1"}} # New thread ID

# --- call_summarizer function ---
def call_summarizer(input_text: str, summarizer_graph = summarizer_graph, config = config):
    """Invokes the summarizer graph, attempts to parse structured output."""
    if not summarizer_graph:
        print("Error: Summarizer graph is not initialized.")
        return None

    agent_input = {"messages": [HumanMessage(content=input_text)]}
    summary_output = None # Initialize variable to store the final summary data

    try:
        result = summarizer_graph.invoke(agent_input, config=config)

        # Extract final answer (attempt to parse JSON from the last AI message)
        if isinstance(result, dict) and "messages" in result:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content') and isinstance(last_message.content, str):
                try:
                    # Attempt to parse the string content as JSON
                    parsed_json = json.loads(last_message.content)
                    # Validate if it matches the Summary structure (optional but good)
                    if all(key in parsed_json for key in ["what", "who", "why", "what_to_do"]):
                         summary_output = parsed_json # Store the parsed dictionary
                    else:
                         print("Warning: Parsed JSON does not match expected Summary fields. Storing raw string.")
                         summary_output = last_message.content # Fallback to string
                except json.JSONDecodeError:
                    print("Warning: Could not parse final agent message as JSON. Storing raw string.")
                    summary_output = last_message.content # Fallback to raw string
                except Exception as parse_exc:
                    print(f"Warning: Error processing final message content: {parse_exc}. Storing raw string.")
                    summary_output = last_message.content # Fallback to raw string
            else:
                 print("Warning: Last message content is not a string. Storing raw result.")
                 summary_output = result # Fallback for unexpected message content
        else:
            print("Warning: Agent result format unexpected. Storing raw result.")
            summary_output = result # Fallback for unexpected result format

        # print(f"------------ Raw result from agent: {result}") # Debugging
        return summary_output # Return the parsed dict or fallback string/dict

    except Exception as e:
        print(f"Error during agent invocation: {e}")
        # traceback.print_exc()
        return None # Indicate failure

# --- Main execution block ---
def run_summarization(input_file: str, output_file: str, examples_to_process: int = -1):
    """Loads data, runs summarization, and saves results to a JSON file."""
    if not summarizer_graph:
        print("Exiting: Summarizer agent could not be initialized.")
        return

    # --- Load Input Data ---
    try:
        print(f"Loading data from: {input_file}")
        with open(input_file, "r", encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"Error: Expected root of JSON file '{input_file}' to be a list.")
            return
        print(f"Loaded {len(data)} entries.")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file}'.")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading data: {e}")
        return

    output_data = [] # List to store results with summaries
    processed_count = 0

    # Determine how many examples to process
    num_to_process = len(data) if examples_to_process < 0 else min(examples_to_process, len(data))
    print(f"Processing up to {num_to_process} examples...")

    # --- Process Data ---
    for i, example in enumerate(data):
        if i >= num_to_process:
            break

        print(f"\n===== Processing Example {i+1}/{num_to_process} =====")
        # Create a deep copy to store results without modifying original data
        output_example = copy.deepcopy(example)

        # Process Moving Brief
        if 'moving_brief' in output_example and 'brief_arguments' in output_example['moving_brief']:
            print(f"  Processing {len(output_example['moving_brief']['brief_arguments'])} moving brief arguments...")
            for j, content_item in enumerate(output_example['moving_brief']['brief_arguments']):
                if 'content' in content_item and content_item['content']:
                    argument_text = content_item['content']
                    try:
                        prompt_value = summarizer_prompt.invoke({"argument": argument_text})
                        input_string = prompt_value.to_string()
                        summary_result = call_summarizer(input_string, summarizer_graph=summarizer_graph, config=config)

                        # Add summary to the output structure
                        content_item['summary'] = summary_result if summary_result else "Summarization Failed"
                        print(f"    Processed moving arg {j+1}")

                    except Exception as e:
                        print(f"Error processing moving brief argument {j+1}: {e}")
                        content_item['summary'] = f"Error during processing: {e}"
                else:
                    content_item['summary'] = "No content found" # Handle missing content

        # Process Response Brief
        if 'response_brief' in output_example and 'brief_arguments' in output_example['response_brief']:
             print(f"  Processing {len(output_example['response_brief']['brief_arguments'])} response brief arguments...")
             for k, content_item in enumerate(output_example['response_brief']['brief_arguments']):
                 if 'content' in content_item and content_item['content']:
                     argument_text = content_item['content']
                     try:
                         prompt_value = summarizer_prompt.invoke({"argument": argument_text})
                         input_string = prompt_value.to_string()
                         summary_result = call_summarizer(input_string, summarizer_graph=summarizer_graph, config=config)

                         # Add summary to the output structure
                         content_item['summary'] = summary_result if summary_result else "Summarization Failed"
                         print(f"    Processed response arg {k+1}")

                     except Exception as e:
                         print(f"Error processing response brief argument {k+1}: {e}")
                         content_item['summary'] = f"Error during processing: {e}"
                 else:
                     content_item['summary'] = "No content found" # Handle missing content

        output_data.append(output_example)
        processed_count += 1

    # --- Save Output Data ---
    try:
        print(f"\n===== Writing {processed_count} processed examples to {output_file} =====")
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=4) # Use indent for readability
        print("Output successfully saved.")
    except Exception as e:
        print(f"Error writing output JSON to '{output_file}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize arguments in legal briefs and save to JSON.")
    parser.add_argument("input_json", help="Path to the input JSON file (e.g., stanford_hackathon_brief_pairs_clean.json).")
    parser.add_argument("output_json", help="Path to the output JSON file (e.g., stanford_hackathon_brief_pairs_with_summary.json).")
    parser.add_argument("-n", "--num_examples", type=int, default=-1,
                        help="Number of examples to process from the input file (-1 for all).")

    args = parser.parse_args()

    run_summarization(args.input_json, args.output_json, args.num_examples)
