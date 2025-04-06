# this file allows us to quickly test our ReAct agent and gives us minimal control over its inner workings and interperetability
# TODO: make this better!

from models import high_thinking_model
from tools import tools
# does printstream work for this usecase? 
# otherwise, is there anyway to interact with the copilotkit?
from langgraph.prebuilt import create_react_agent 
from langgraph.checkpoint.memory import MemorySaver

# It would be best to find a way to scrape away these indecies. Perhaps we don't give the headings in at all. 
system_prompt = """You are an legal assistant. You will be given an argument, heading and a previous summary of the contents 
(a summary that isn't neccisarily trustworthy) of both an argument and a series of possible response argements. 
Your goal is to match the input argument to it's most probable responses. Sometimes there is no response 
and sometimes there is several. But, for each input you must look at each output to see if it would make 
sense as a response to the input. Also, there may be indexing in front of the heading. Please 
ignore these as they are placed there for reasons unrelated to our problem. """

# TODO: force the agent to output in our certain way. A list of pairs of matches with the input. Format '[[argument, response1],[argument, response2],...]'
agent_graph = create_react_agent(model = high_thinking_model, tools = tools, checkpointer=MemorySaver()) 

config = {"configurable": {"thread_id": "thread-1"}}

def print_stream(graph, inputs, config):
    for s in graph.stream(inputs, config, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

def call_agent(input: str, agent_graph = agent_graph):
    return agent_graph.invoke(input)

def print_agent(input: str, agent_graph = agent_graph, config = config):
    print_stream(input, agent_graph, config) 