import os
from dotenv import load_dotenv
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_community import GoogleSearchAPIWrapper
from copilotkit import CopilotKitState # extends MessagesState
from copilotkit.langgraph import copilotkit_emit_state # we don't really need copilot for this part. Let's generate a report using memory instead.

# we should only load these in one file. 
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GENERAL_SEARCH_KEY = os.getenv("GOOGLE_GENERAL_SEARCH")
gen_search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GENERAL_SEARCH_KEY)

# Define models
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-preview-03-25", api_key=GOOGLE_API_KEY)
#thinking_model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", api_key=GOOGLE_API_KEY)
#low_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", api_key=GOOGLE_API_KEY)

# these classes may be used in other files. We should make a general file we can import to wherever
class Summary(BaseModel): # what does the base model class actually do?
    "Summary to assist in decision making"
    what: str = Field(description = "What is this argument really saying?", max_length=100)
    who: str = Field(description = "Who is the speaker and who is their likely audiance?", max_length=100)
    why: str = Field(description = "What is the likely motive behind this argument?", max_length=100)
    # what_to_do: str = Field(description = "What is the most effective way to respond to this argument?", max_length=100)

class AgentState(CopilotKitState): # this allows for intermittant emissions of information
    messages: List[Dict[str, str]]

# this is our summary tool. We should define more tools and place these in a shared tools file 
# so that every agent and graph can use all our tools. What does this code actually accomplish?
@tool()
def SummarizeTool(summary: str): # pylint: disable=invalid-name,unused-argument
    """
    Summarize the argument and title. Ignore whatever indexing may be included in either the title or argument.
    Make sure that the summary is complete, anwsering all questions totally and consisely.
    """

# this is the driving function in this file. But, we have to remember that this 
# summarizer is independent
async def summarize_node(state: AgentState, config: RunnableConfig):
    await copilotkit_emit_state(config, state)

    system_message = """Your goal is to summarize the argument you have been given.
    You have been given access to google search and wikipedia to help you in accomplishing this"""
    
    response = await model.bind_tools(
        [SummarizeTool], 
        "SummarizeTool"
    ).ainvoke([
        HumanMessage(content=system_message),
    ], config)

    return {"anwser":response.tool_calls[0]["args"]}