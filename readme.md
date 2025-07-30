# Argument Link

Argument Link is a Python-based legal argument analysis system that leverages advanced AI agents to automatically identify and link responses to legal arguments. Originally developed for the Stanford LLM x Law Hackathon 2025, it uses Google's Gemini models and LangChain to provide intelligent argument matching with verification capabilities.

## Features

- **Intelligent Argument Summarization**: Automatically analyzes legal arguments to extract key content, speaker/audience context, and underlying motives
- **Dynamic Response Matching**: Uses ReAct agents to identify which response arguments correspond to specific moving arguments
- **Verification System**: Validates argument-response links to ensure accuracy and relevance
- **Multi-Model Architecture**: Employs tiered Gemini models for optimal performance across different complexity levels
- **Research Integration**: Built-in tools for Google Search, Federal Register search, Wikipedia, and web browsing
- **Structured Output**: Pydantic models ensure consistent data formatting throughout the pipeline

## Installation

Clone the repo:
```bash
git clone https://github.com/aceniccola/arg_link.git
cd arg_link
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For development, you can also install the package in editable mode:

```bash
pip install -e .
```

Set up your environment variables in a `.env` file:

```bash
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_GENERAL_SEARCH=your_google_search_engine_id
GOOGLE_FEDERAL_SEARCH=your_federal_search_engine_id
```

## Usage

### Basic Usage

**Quick Start:**
```bash
python run.py
```

**Or process a set of legal brief pairs programmatically:**

```python
import json
from src.main import main

# Load your legal brief data
with open("data/stanford_hackathon_brief_pairs_clean.json", "r") as f:
    data = json.load(f)

# Process all examples to find argument links
all_links = main(data)
print(all_links)
```

### Individual Components

**Argument Summarization:**
```python
from src.prebuilt_summarizer import call_summarizer
from src.main import summarizer_prompt

# Summarize a single argument
argument_text = "Your legal argument here"
prompt = summarizer_prompt.invoke({"argument": argument_text})
summary = call_summarizer(prompt)
```

**Response Matching:**
```python
from src.prebuilt_agent import call_agent
from src.main import agent_prompt

# Find responses for an argument
argument = "Your moving argument"
responses = {"brief_arguments": [...]}  # Response brief structure
prompt = agent_prompt.invoke({"argument": argument, "responses": responses})
matches = call_agent(prompt)
```

**Link Verification:**
```python
from src.verifier_agent import call_agent as verify_agent
from src.main import verifier_prompt

# Verify an argument-response link
argument = "Original argument"
response = "Proposed response"
prompt = verifier_prompt.invoke({"argument": argument, "response": response})
verification = verify_agent(prompt)
```

## Data Format

The system expects JSON input with the following structure:

```json
{
  "moving_brief": {
    "brief_arguments": [
      {
        "content": "Argument text here",
        "heading": "Optional heading"
      }
    ]
  },
  "response_brief": {
    "brief_arguments": [
      {
        "content": "Response argument text",
        "heading": "Optional heading"
      }
    ]
  },
  "true_links": [
  ]
}
```

## Project Structure

```
arg_link/
├── src/                    # Core application modules
│   ├── __init__.py
│   ├── main.py            # Main orchestration pipeline
│   ├── models.py          # Gemini model configurations
│   ├── prebuilt_agent.py  # Response matching agent
│   ├── prebuilt_summarizer.py  # Argument summarization
│   ├── verifier_agent.py  # Link verification system
│   ├── tools.py           # Research tools integration
│   └── utils.py           # Utility functions
├── data/                   # Data files
│   ├── input.json
│   ├── stanford_hackathon_brief_pairs.json
│   └── stanford_hackathon_brief_pairs_clean.json
├── scripts/                # Utility and evaluation scripts
│   ├── a.py
│   ├── evaluate.py
│   ├── final_summarizer.py
│   ├── graph.py
│   └── summarizer.py
├── frontend/               # Web interface
│   └── argument_link_frontend.html
├── docs/                   # Documentation
│   ├── Brief Pair Argument Counter Argument Linking.pdf
│   └── BLP HACKATHON IP RELEASE AND ASSIGNMENT Terms.pdf
├── run.py                  # Simple runner script
├── setup.py                # Package setup configuration
├── requirements.txt
└── readme.md
```

## Architecture

- **Models** (`src/models.py`): Three-tier Gemini model configuration for different complexity levels
- **Tools** (`src/tools.py`): Research tools including Google Search, Federal Register, Wikipedia, and web browsing
- **Summarizer** (`src/prebuilt_summarizer.py`): ReAct agent for argument analysis and summarization
- **Matcher** (`src/prebuilt_agent.py`): Core agent for identifying argument-response relationships
- **Verifier** (`src/verifier_agent.py`): Validation system for ensuring link accuracy
- **Main Pipeline** (`src/main.py`): Orchestrates the complete argument linking workflow

## Future Development

The project is actively being developed with planned improvements including:

- **Output Format Standardization**: Consistent structured outputs across all agents
- **Enhanced Verification**: Improved boolean verification system with feedback loops
- **Performance Optimization**: Multi-tier model usage for better speed/accuracy balance
- **Web Interface**: GUI-based interface for easier interaction
- **API Endpoints**: RESTful API for integration with other legal tools
- **Evaluation Metrics**: Comprehensive testing and accuracy measurement systems
