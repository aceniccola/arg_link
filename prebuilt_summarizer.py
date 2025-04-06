# TODO: make this better!
import json
from models import high_thinking_model
from tools import tools
# does printstream work for this usecase? 
# otherwise, is there anyway to interact with the copilotkit?
from langchain_core.prompts import PromptTemplate
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
    print(f"Graph type: {type(graph)}")  # Debugging output
    if not hasattr(graph, 'stream'):
        raise ValueError("The provided graph does not have a 'stream' method.")
    
    for s in graph.stream(inputs, config, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

def call_summarizer(input: str, summarizer_graph = summarizer_graph, config = config):
    return summarizer_graph.invoke(input, config=config)

def print_summarizer(input: str, summarizer_graph = summarizer_graph, config = config):
    print_stream(summarizer_graph, input, config) 

if __name__ == "__main__":
    with open("stanford_hackathon_brief_pairs_clean.json", "r") as f:
        data = json.load(f)
    for example in data:
        print("-------------------------------- moving brief --------------------------------")
        for content in example['moving_brief']['brief_arguments']:
            # Convert StringPromptValue to string using .to_string()
            prompt_value = summarizer_prompt.invoke({"argument": content['content']})
            call_summarizer(prompt_value, summarizer_graph=summarizer_graph)
        print("-------------------------------- response brief --------------------------------")
        for content in example['response_brief']['brief_arguments']:
            # Convert StringPromptValue to string using .to_string()
            prompt_value = summarizer_prompt.invoke({"argument": content['content']})
            call_summarizer(prompt_value, summarizer_graph=summarizer_graph)