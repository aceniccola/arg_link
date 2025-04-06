# TODO: make this better!
import json
from models import high_thinking_model # Assuming this is correctly configured Gemini model
from tools import tools # Assuming these are defined LangChain tools
# does printstream work for this usecase?
# otherwise, is there anyway to interact with the copilotkit?
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage # Import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

#DONE: to get rid of the heading indecies, try using the regex gemini as your first pass.
# You should have enough to get by with it.
# --- OR ---
#TODO: Perhaps we don't give the headings in at all.
summarizer_prompt = PromptTemplate.from_template( """ Your goal is to summarize an argument with the following 3 conditions:
1. What is this argument really saying
2. Who is the speaker and who is their likely audiance?
3. What is the likely motive behind this argument?

Here is the argument: {argument} """)

class Summary(BaseModel):
    "Summary to assist in decision making"
    what: str = Field(description = "What is this argument really saying?", max_length=100)
    who: str = Field(description = "Who is the speaker and who is their likely audiance?", max_length=100)
    why: str = Field(description = "What is the likely motive behind this argument?", max_length=100)
    what_to_do: str = Field(description = "What is the most effective way to respond to this argument?", max_length=100)

# TODO: force the agent to output in our certain way. A dictionary of anwsers to the questions asked. Format '{q1:a1,q2:a2,...}'
# --- OR ---
# TODO: Even better: we can have our model respond with something in the Summaary class
summarizer_graph = create_react_agent(model = high_thinking_model, tools = tools, checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "thread-1"}}

def print_stream(graph, inputs, config):
    """Streams output from a LangGraph graph."""
    print(f"Graph type: {type(graph)}")
    if not hasattr(graph, 'stream'):
        print("Warning: The provided graph does not have a 'stream' method. Trying invoke.")
        # Fallback or error handling if stream isn't available
        try:
            result = graph.invoke(inputs, config)
            print("Invoke Result:")
            # Attempt to pretty print if possible, otherwise just print
            if hasattr(result, 'pretty_print'):
                 result.pretty_print()
            elif isinstance(result, dict) and "messages" in result:
                 result["messages"][-1].pretty_print()
            else:
                 print(result)
        except Exception as e:
            print(f"Error during fallback invoke: {e}")
        return

    print("Streaming output:")
    try:
        # Ensure inputs are in the correct dictionary format for stream as well
        if not isinstance(inputs, dict) or "messages" not in inputs:
             print("Warning: Input to stream might not be in the expected format.")
             # Attempt basic conversion if it's just a string
             if isinstance(inputs, str):
                 inputs = {"messages": [HumanMessage(content=inputs)]}
             else:
                 print("Cannot automatically convert input for streaming. Proceeding anyway.")


        for s in graph.stream(inputs, config, stream_mode="values"):
            # Access the latest message, structure might vary slightly
            # based on graph specifics, adjust if needed.
            # Common patterns: s['messages'][-1] or s['agent']['messages'][-1]
            message = None
            if "messages" in s and s["messages"]:
                 message = s["messages"][-1]
            elif "agent" in s and isinstance(s["agent"], dict) and "messages" in s["agent"] and s["agent"]["messages"]:
                 message = s["agent"]["messages"][-1]
            else:
                 # Print the whole chunk if message structure is unexpected
                 print("Streaming chunk:", s)
                 continue # Skip to next chunk

            if message:
                 if hasattr(message, 'pretty_print'):
                     message.pretty_print()
                 else:
                     print(message) # Fallback print
            else:
                print("Empty message in stream chunk:", s)

    except Exception as e:
        print(f"Error during streaming: {e}")


# --- MODIFIED FUNCTION ---
def call_summarizer(input_text: str, summarizer_graph = summarizer_graph, config = config):
    """
    Invokes the summarizer graph with the input text formatted correctly.

    Args:
        input_text: The string content to be summarized.
        summarizer_graph: The LangGraph agent executor.
        config: The configuration dictionary for the graph invocation.

    Returns:
        The result from graph.invoke().
    """
    # Wrap the input string in the expected dictionary format
    agent_input = {"messages": [HumanMessage(content=input_text)]}
    print(f"Invoking agent with input: {agent_input}") # Debug print
    return summarizer_graph.invoke(agent_input, config=config)

# --- MODIFIED FUNCTION ---
def print_summarizer(input_text: str, summarizer_graph = summarizer_graph, config = config):
    """
    Streams the summarizer graph output with the input text formatted correctly.

    Args:
        input_text: The string content to be summarized.
        summarizer_graph: The LangGraph agent executor.
        config: The configuration dictionary for the graph streaming.
    """
    # Wrap the input string in the expected dictionary format
    agent_input = {"messages": [HumanMessage(content=input_text)]}
    print(f"Streaming agent with input: {agent_input}") # Debug print
    print_stream(summarizer_graph, agent_input, config)

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

    # Limit the number of examples for initial testing
    examples_to_process = 1 # Adjust as needed

    for i, example in enumerate(data):
        if i >= examples_to_process:
            break

        print(f"\n===== Processing Example {i+1} =====")

        if 'moving_brief' in example and 'brief_arguments' in example['moving_brief']:
            print("\n--- Moving Brief Arguments ---")
            for j, content in enumerate(example['moving_brief']['brief_arguments']):
                 if 'content' in content:
                     print(f"\nArgument {j+1}:")
                     try:
                         # --- MODIFIED CALL ---
                         # 1. Invoke the prompt template to get the PromptValue
                         prompt_value = summarizer_prompt.invoke({"argument": content['content']})
                         # 2. Convert PromptValue to string for the agent input
                         input_string = prompt_value.to_string()
                         # 3. Call the summarizer function with the string
                         result = call_summarizer(input_string, summarizer_graph=summarizer_graph, config=config)
                         # Optional: Print result if not using streaming, or if call_summarizer doesn't print
                         # print("Summarizer Result:", result)
                         # Or use the streaming version:
                         # print_summarizer(input_string, summarizer_graph=summarizer_graph, config=config)

                     except Exception as e:
                         print(f"Error processing moving brief argument {j+1}: {e}")
                         # Optionally add more detailed error logging or continue to the next item
                         # import traceback
                         # traceback.print_exc()
                 else:
                     print(f"Skipping moving brief argument {j+1} due to missing 'content' key.")

        if 'response_brief' in example and 'brief_arguments' in example['response_brief']:
            print("\n--- Response Brief Arguments ---")
            for k, content in enumerate(example['response_brief']['brief_arguments']):
                 if 'content' in content:
                     print(f"\nArgument {k+1}:")
                     try:
                         # --- MODIFIED CALL ---
                         prompt_value = summarizer_prompt.invoke({"argument": content['content']})
                         input_string = prompt_value.to_string()
                         result = call_summarizer(input_string, summarizer_graph=summarizer_graph, config=config)
                         # print("Summarizer Result:", result)
                         # Or use the streaming version:
                         # print_summarizer(input_string, summarizer_graph=summarizer_graph, config=config)

                     except Exception as e:
                         print(f"Error processing response brief argument {k+1}: {e}")
                 else:
                     print(f"Skipping response brief argument {k+1} due to missing 'content' key.")

    print("\n===== Processing Complete =====")
