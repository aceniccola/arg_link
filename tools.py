import os
from dotenv import load_dotenv
from langchain_core.tools import Tool
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,  # A synchronous browser is available
)

# load environment variables and initialize search tools
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GENERAL_SEARCH_KEY = os.getenv("GOOGLE_GENERAL_SEARCH")
FEDERAL_SEARCH_KEY = os.getenv("GOOGLE_FEDERAL_SEARCH")
gen_search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GENERAL_SEARCH_KEY)
fed_search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=FEDERAL_SEARCH_KEY)

wiki_tool = WikipediaQueryRun(
    name="Wikipedia", 
    api_wrapper=WikipediaAPIWrapper(), 
    description="""Should be able to have great success obtaining information about a wide range 
    of topics inside and outside of the domain of law."""
    )

gen_search_tool = Tool(
    name = "Google Search",
    description = "Search Google for any recent relevent results.",
    func = gen_search.run
)

fed_search_tool = Tool(
    name = "Federal Search",
    description = """Search the Federal Register for any recent relevent results. 
    These results are relevent to possible legislation, regulations, and other federal documents.""",   
    func = fed_search.run
)

async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)

# add the wiki and google search tools to the search toolkit 
tools = toolkit.get_tools()
tools.extend([wiki_tool, gen_search_tool, fed_search_tool])

# Fix invalid function names for Gemini compatibility
for i, tool in enumerate(tools):
    # Check if the name is valid for Gemini API (must start with letter/underscore and contain only alphanumeric, underscores, dots, or dashes)
    if not tool.name or not (tool.name[0].isalpha() or tool.name[0] == '_'):
        # Rename tool with invalid name
        old_name = tool.name
        tool.name = f"tool_{i}" if not tool.name else f"tool_{tool.name.replace(' ', '_')}"
        print(f"Renamed tool '{old_name}' to '{tool.name}' for API compatibility")
    
    # Replace any invalid characters
    tool.name = ''.join(c for c in tool.name if c.isalnum() or c in ['_', '.', '-'])

if __name__ == "__main__":
    print("\nThe following are the tools that we can use in this project:\n")
    for tool in tools:
        print(tool.name)
        print(tool.description)
        print("----------------------------------------------------------------")



