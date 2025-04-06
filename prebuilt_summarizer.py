# TODO: make this better!

from models import high_thinking_model
from tools import tools
# does printstream work for this usecase? 
# otherwise, is there anyway to interact with the copilotkit?
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver 

#TODO: to get rid of the heading indecies, try using the regex gemini as your first pass. 
# You should have enough to get by with it.
# --- OR ---
#TODO: Perhaps we don't give the headings in at all. 
system_prompt = """You are an legal assistant. You will be given an argument and a heading and 
your task will be to summarize the argument in the following style """

# TODO: force the agent to output in our certain way. A dictionary of anwsers to the questions asked. Format '{q1:a1,q2:a2,...}'
# --- OR ---
# TODO: Even better: we can have our model respond with something in the Summaary class
summarizer_graph = create_react_agent(model = high_thinking_model, tools = tools, checkpointer=MemorySaver()) 

config = {"configurable": {"thread_id": "thread-1"}}

def print_stream(graph, inputs, config):
    for s in graph.stream(inputs, config, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

def call_summarizer(input: str, summarizer_graph = summarizer_graph):
    return summarizer_graph.invoke(input)

def print_summarizer(input: str, summarizer_graph = summarizer_graph, config = config):
    print_stream(input, summarizer_graph, config) 