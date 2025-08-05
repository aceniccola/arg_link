# Project Roadmap: Argument Link

This document outlines the planned future development for the Argument Link project. The goal is to evolve the current proof-of-concept into a robust, production-ready legal analysis tool.

## Key Development Areas

The following sections detail the major initiatives planned for the project, building upon the existing architecture and addressing the current `TODO` items in the codebase.

---

### 1. Agent Output Standardization

**Goal:** Ensure all agents (`summarizer`, `matcher`, `verifier`) return consistent, structured, and machine-readable output.

**Tasks:**
- **Define Pydantic Models:** Implement strict Pydantic models for all agent return types.
- **Enforce Output Format:** Use LangChain's output parsing or function calling features to force agents to adhere to the defined schemas.
- **Eliminate String Parsing:** Refactor any code that relies on parsing unstructured string outputs from the LLMs.

**Benefit:** Greatly improves system reliability, enables easier data exchange between components, and simplifies testing.

---

### 2. Enhanced Verification System

**Goal:** Create a closed-loop verification system that provides feedback and can self-correct.

**Tasks:**
- **Implement Boolean Verifier:** Refine the `verifier_agent` to return a simple `True`/`False` or a confidence score.
- **Introduce a Reflection Agent:** Develop a "reflection" or "critic" agent that can analyze the matcher's output and the verifier's feedback to provide actionable suggestions.
- **Create a Correction Loop:** Modify the main pipeline in `src/main.py` to re-run the matching process with the feedback from the reflection agent if verification fails.

**Benefit:** Increases the accuracy of argument links and moves towards a more autonomous, self-correcting system.

---

### 3. Performance & Cost Optimization

**Goal:** Optimize the model architecture for a better balance between performance, cost, and accuracy.

**Tasks:**
- **Tiered Model Strategy:** Implement logic to use smaller, faster models (like `gemini-flash`) for simpler tasks (e.g., initial summarization) and reserve the more powerful models (e.g., `gemini-pro`) for complex reasoning and verification.
- **Implement Caching:** Integrate a caching mechanism (e.g., `langchain.cache.InMemoryCache`) to avoid redundant API calls for identical inputs.
- **Batch Processing:** Refactor the data processing loop to handle briefs in batches where possible.

**Benefit:** Reduces API costs, lowers latency, and makes the system more scalable.

---

### 4. Interactive Web Interface

**Goal:** Develop the existing `argument_link_frontend.html` into a fully interactive web application.

**Tasks:**
- **Backend API:** Build a simple backend using FastAPI or Flask to serve the frontend and expose the core logic.
- **Frontend Framework:** Port the HTML to a modern frontend framework like React or Vue.js for better state management.
- **Real-time Updates:** Use WebSockets or server-sent events to provide real-time updates as the analysis is being performed.
- **Visualization:** Create visualizations for the argument links, perhaps using a graph library like D3.js or Vis.js.

**Benefit:** Makes the tool accessible to non-technical users and provides a more intuitive way to explore the results.

---

### 5. API for Integration

**Goal:** Expose the core functionality through a well-documented RESTful API.

**Tasks:**
- **FastAPI Endpoints:** Create API endpoints in `src/main.py` (or a new `api.py`) for summarizing text, matching arguments, and processing full briefs.
- **API Documentation:** Auto-generate API documentation using FastAPI's built-in OpenAPI/Swagger support.
- **Authentication:** Add a simple API key authentication mechanism.

**Benefit:** Allows Argument Link to be integrated into other legal tech workflows and applications.

---

### 6. Comprehensive Evaluation Suite

**Goal:** Build a robust evaluation system to measure and track the system's accuracy.

**Tasks:**
- **Refine `scripts/evaluate.py`:** Expand the script to calculate precision, recall, and F1-score.
- **Develop Test Sets:** Curate and expand the `true_links` in the data files to create a comprehensive test suite.
- **Automated Testing:** Set up a CI/CD pipeline (e.g., using GitHub Actions) to run the evaluation script automatically on each commit.

**Benefit:** Provides quantitative metrics for system performance and prevents regressions during development.
