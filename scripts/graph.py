# Revised LangGraph Flow using Gemini Function Calling

# --- Imports ---
import json
from utils import striptojson
from tools import tools 
from models import thinking_model
from typing import List, Dict, Any, Optional
from langchain_core.messages import (
    ToolMessage,
    AIMessage,
    HumanMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode  # Use prebuilt ToolNode
from copilotkit import CopilotKitState # Extends MessagesState
from copilotkit.langgraph import copilotkit_emit_state

# --- Models ---
llm = thinking_model.bind_tools(tools)

# --- State Definition ---
class ArgumentAnalysisState(CopilotKitState):
    """
    Represents the state of the argument analysis graph.
    Inherits from CopilotKitState for UI integration.

    Attributes:
        messages: The list of messages exchanged.
        argument: The original argument text.
        responses: The list of potential response texts or dicts.
        verified_responses: List of responses verified as relevant by the agent.
        tool_status: Dictionary to track tool usage for UI feedback.
    """
    argument: Dict[str, str] # Assuming argument is a dict with keys "heading", "content", and "summary"
    responses: List[Dict[str, str]] 
    verified_responses: Optional[List[Dict[str, str]]] = None
    tool_status: Dict[str, Any] = {} 

# --- Nodes ---
def argument_analyzer_node(state: ArgumentAnalysisState, config: RunnableConfig):
    """
    Analyzes the argument, identifies relevant responses, and potentially calls tools. 
    Sends the response to either the verifier or back to the agent.
    """
    print("--- Entering Argument Analyzer Node ---")
    argument = state["argument"]
    responses = state["responses"]

    # Create a more structured prompt if needed, maybe sending responses as a numbered list
    response_content_list = [f"{idx + 1}. {resp.get('heading', '')}: {resp.get('content', '')}. {resp.get('summary','')}" for idx, resp in enumerate(responses)]
    prompt = f"""You are an expert legal analyst specializing in argument-response mapping.
Your task is to analyze the provided 'Argument' and determine which of the 'Potential Responses' directly address or counter the points made in the 'Argument'.

Argument:
---
{argument}
---

Potential Responses:
---
{chr(len(responses)).join(response_content_list)}
---

Instructions:
1. Carefully read the Argument and each Potential Response using the summaries to assist you decision making.
2. Identify the responses relevent to the argument. These should be the responses that directly address or counter the points made in the argument. If you have any question regarding an argument or a response, use tools to solve it.
3. If you need external information (e.g., definitions, case law details) to make a better judgment, use the available tools.
4. Output your final answer as a JSON object containing a single key "relevant_response_indices" with a list of the 1-based indices of the relevant responses. Remember to use no additional newlines. Example: {{"relevant_response_indices": [1, 3]}}
5. If no responses are relevant, return: {{"relevant_response_indices": []}}
6. If you use a tool, the next step will incorporate the tool's output. You will then be asked to provide the final JSON response based on the combined information. Do not output the JSON structure if you are calling a tool. Instead, explain why you need the tool.

You must call tools to answer the question.
"""
    # Append the user prompt to the message history if it's not already the last message
    # Or construct the message list dynamically for each invocation if preferred
    current_messages = state["messages"] + [HumanMessage(content=prompt)]

    # Invoke the LLM (which has tools bound)
    response_message = llm.invoke(current_messages, config=config)
    print(f"--- LLM Response: {response_message} ---")

    
    # turn the response into a json if it's not already
    messageio = response_message.content
    if type(messageio) is str: # is not list ; maybe 
        messageio = striptojson(messageio)

    # Store the identified relevant responses (if no tool call)
    verified_responses_list = []
    if not response_message.tool_calls:
        try:
            output_json = json.loads(messageio)
            relevant_indices = output_json.get("relevant_response_indices", [])
            verified_responses_list = [responses[i - 1] for i in relevant_indices if 0 < i <= len(responses)]
            print(f"--- Verified Responses (Indices: {relevant_indices}): {verified_responses_list} ---")
        except json.JSONDecodeError:
            print("--- Error: LLM did not return valid JSON. ---")
            pass 

    return {
        "messages": [response_message], # Return only the new message for LangGraph state update
        "verified_responses": verified_responses_list if not response_message.tool_calls else state.get("verified_responses"), # Update only if final response
        "tool_status": {} # Reset tool status unless a tool is called
        }

# Use LangGraph's prebuilt ToolNode, but wrap it for CopilotKit state emission
# NOTE: LangGraph's ToolNode executes tools based on the *last* AIMessage.
# Our agent node returns *only* the new AIMessage, so this works.
tool_node = ToolNode(tools)
def wrapped_tool_node(state: ArgumentAnalysisState, config: RunnableConfig):
    """Wraps the LangGraph ToolNode to emit CopilotKit state updates."""
    print("--- Entering Tool Node ---")
    last_message = state["messages"][-1]
    tool_status_updates = {}

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Emit "processing" state before execution
        # Note: We don't know *which* tool will be called yet by ToolNode logic easily
        tool_status_updates = {"status": "processing", "tool": "unknown"}
        copilotkit_emit_state(config, {"tool_status": tool_status_updates})

        # Call the actual ToolNode
        output_state = tool_node.invoke(state) # ToolNode expects the full state

        # Emit "complete" or "error" state after execution
        # Check the output for ToolMessages or errors
        last_output_message = output_state["messages"][-1]
        if isinstance(last_output_message, ToolMessage):
             tool_status_updates = {"status": "complete", "tool": last_output_message.name}
        else:
             # Basic error assumption - enhance if ToolNode provides better error info
             tool_status_updates = {"status": "error", "tool": "unknown", "error": "Tool execution failed or produced unexpected output"}

        copilotkit_emit_state(config, {"tool_status": tool_status_updates})
        print(f"--- Tool Node Output State: {output_state} ---")
        return {**output_state, "tool_status": tool_status_updates} # Return the result from ToolNode + status

    else:
        # Should not happen if routing is correct, but handle defensively
        print("--- Tool Node Skipped (No tool calls in last message) ---")
        return {"tool_status": {"status": "skipped"}}

# for now, we just return the verified responses, but the intended functionality is to verify
# the responses and return the final answer. A double check of the responses using less tools and 
# yet able to send the response back as invalid or send it as output.
def verifier_node(state: ArgumentAnalysisState, config: RunnableConfig):
    """
    Verifies the responses and returns the final answer.
    """
    print("--- Entering Verifier Node ---")
    verified_responses = state["verified_responses"]
    return {"verified_responses": verified_responses}

# --- Edges ---
def should_use_tools(state: ArgumentAnalysisState) -> str:
    """Determines whether the agent decided to call a tool."""
    print("--- Evaluating Edges ---")
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print("--- Decision: Use Tools ---")
        return "use_tools"
    print("--- Decision: End ---")
    return "__end__" # Use LangGraph's END constant

# --- Graph Definition ---
graph_builder = StateGraph(ArgumentAnalysisState)
graph_builder.add_node("analyzer", argument_analyzer_node)
graph_builder.add_node("tools", wrapped_tool_node) # Use the wrapped tool node
graph_builder.set_entry_point("analyzer")

# Conditional edge: After analyzer, check if tools are needed
graph_builder.add_conditional_edges(
    "analyzer",
    should_use_tools,
    {
        "use_tools": "tools",
        "__end__": END,
    },
)

# After tools are executed, return to the analyzer to process results
graph_builder.add_edge("tools", "analyzer")

# Compile the graph
app = graph_builder.compile()

# --- Example Invocation ---
# (Assuming 'argument' and 'responses' variables are loaded as in the original script)

with open("input.json", "r") as file:
    # data = file.read() # Incorrect: reads the file as a single string
    data = json.load(file) # Correct: parses the entire JSON file content

for ex in data:
    # for example
    movings = ex['moving_brief']
    responses = ex['response_brief']
    responses_list = responses["brief_arguments"]
    # Iterate through the list of arguments within moving_brief
    for moving_arg in movings["brief_arguments"]:
        heading = moving_arg['heading'] # Access directly from the argument dict
        content = moving_arg['content']
        summary = moving_arg.get('summary', '') # Use .get for summary as it might be optional. why though use 'get' here

        initial_state = ArgumentAnalysisState(
            messages = [],
            argument = {'heading': heading, 'content': content, 'summary': str(summary)},
            responses = responses_list,
            verified_responses = None,
            tool_status = {}
        )

        # Define a config, needed for CopilotKit state emission
        # In a real app, this config would be passed down from the request handler
        config: RunnableConfig = {"configurable": {"user_id": "test-user", "thread_id": "test-thread"}}

        # Invoke the graph
        # Use stream or invoke depending on how you want to get results
        final_state = app.invoke(initial_state, config=config) # TODO: make this work for using tools

        print("\n--- Final State ---")
        print("Verified Pairs: \n")
        
        # for pair in pairs of responses on this moving argument
        for response in final_state.get('verified_responses'):
            with open("dump.txt", "a") as f:
                f.write(f"{[heading, response['heading']]}\n")
                print([heading, response['heading']])
        
# print(f"Full Message History: {final_state.get('messages')}")