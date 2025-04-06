import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Define models
high_thinking_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-preview-03-25", api_key=GOOGLE_API_KEY)
thinking_model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", api_key=GOOGLE_API_KEY)
low_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", api_key=GOOGLE_API_KEY)

""" Was thinking.  if we can't make the basic high model work fast enough to do all the tool uses and also 
make the decision, we can make an ai researching small llm to do the research and send it off to the heavier 
model afterwards in order to make the final decision. Would be nice to have a thinking model do this 
(even if I have no idea how to make them think)"""

"for now we move forward with the large model for each task. And hopefully interpertability is strong enough."