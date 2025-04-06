# this is the implementation of our main graph function without using
# prebuilt agent, and therefore definitely able to be leveraged by
# copilotkit

# first, we import our state submitting information for our ui
from copilotkit import CopilotKitState # extends MessagesState
from copilotkit.langgraph import copilotkit_emit_state

# we also need all our dependencies we built up ourselves
from models import high_thinking_model, thinking_model, low_model
from tools import tools

# next, we import all that is required to build out our graph
from typing import List, Dict, Any, Optional
import re
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

class AgentState(CopilotKitState):
    messages: List[Dict[str, str]]
    # Add a state tracker for tool usage
    tool_usage: Dict[str, Any] = {}

# ---- nodes ----
def agent_node(state: AgentState, config: RunnableConfig):
    messages = state["messages"]
    response = high_thinking_model.ainvoke(messages)
    return {"messages": [*messages, response]}

def verifier_node(state:AgentState, config: RunnableConfig):
    messages = state["messages"]
    response = high_thinking_model.ainvoke(messages)
    return {"messages": [*messages, response]}

def tool_node(state: AgentState, config: RunnableConfig):
    """Process tool actions from the agent's messages and execute the appropriate tool."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Update the state for UI feedback
    state["tool_usage"] = {"status": "processing", "tool": None}
    copilotkit_emit_state(config, state)
    
    # Parse the action and tool input from the last message
    tool_name, tool_input = parse_tool_request(last_message.content)
    
    if not tool_name:
        return {"messages": [*messages, AIMessage(content="No valid tool action detected.")]}
    
    # Update the state with the tool being used
    state["tool_usage"] = {"status": "running", "tool": tool_name}
    copilotkit_emit_state(config, state)
    
    # Execute the tool
    try:
        # Find the requested tool in our tools list
        selected_tool = next((tool for tool in tools if tool.name.lower() == tool_name.lower()), None)
        
        if selected_tool:
            tool_output = selected_tool.invoke(tool_input)
            tool_message = ToolMessage(content=str(tool_output), tool_name=tool_name)
            
            # Update state to show completion
            state["tool_usage"] = {"status": "complete", "tool": tool_name}
            copilotkit_emit_state(config, state)
            
            return {"messages": [*messages, tool_message]}
        else:
            return {"messages": [*messages, AIMessage(content=f"Tool '{tool_name}' not found.")]}
    
    except Exception as e:
        # Handle any errors during tool execution
        error_message = f"Error executing tool '{tool_name}': {str(e)}"
        state["tool_usage"] = {"status": "error", "tool": tool_name, "error": str(e)}
        copilotkit_emit_state(config, state)
        
        return {"messages": [*messages, AIMessage(content=error_message)]}

def parse_tool_request(message_content: str) -> tuple[Optional[str], Optional[str]]:
    """Parse the tool name and input from a message content."""
    # Try to extract structured tool calls using regex patterns
    # Q: must we do it this way? I feel there should be a dimplier way 
    # to ensure the llm gives a structured response.
    # todo 
    action_pattern = r"(action|tool|use)[\s]*:[\s]*([a-zA-Z0-9 ]+)"
    input_pattern = r"(input|query|parameter)[\s]*:[\s]*(.+?)(?=\n|$)"
    
    action_match = re.search(action_pattern, message_content, re.IGNORECASE)
    input_match = re.search(input_pattern, message_content, re.IGNORECASE)
    
    tool_name = action_match.group(2).strip() if action_match else None
    tool_input = input_match.group(2).strip() if input_match else ""
    
    # If no structured format is found but "action" is in the message,
    # try to extract any tool name after "action:"
    if not tool_name and "action" in message_content.lower():
        parts = message_content.split(":")
        if len(parts) > 1:
            tool_name = parts[-1].strip()
            # Use the previous message as input if no specific input was provided
            if not tool_input:
                tool_input = ""  # Default empty input
    
    return tool_name, tool_input

# ---- edges ----
def should_continue(state: AgentState): 
    messages = state["messages"]
    last_message = messages[-1]

    if "action" in last_message.content.lower():
        return "use_tool"
    
    return "continue"

def verify_or_revise(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    if "1" in last_message.content:
        return "verify"
    elif "2" in last_message.content:
        return "revise"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)
graph.add_node("verifier", verifier_node)

graph.set_conditional_edge("agent", should_continue, {
    "use_tool": "tool",
    "continue": "verifier"
})

graph.set_conditional_edge("tool", verify_or_revise, {
    "verify": "verifier",
    "revise": "agent",
    "continue": END
})

graph.add_edge("tool", "agent")

app = graph.compile()

# example invocation
"""inputs = {"messages": [HumanMessage(content="What is the capital of France?")]}
result = app.invoke(inputs)
print(result)

inputs2 = {"messages": [HumanMessage(content="What is the current weather in Paris?")]}
result2 = app.invoke(inputs2)
print(result2)"""