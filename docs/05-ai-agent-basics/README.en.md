# 🤖 AI Agent Interview Guide

> **Difficulty:** ⭐⭐⭐
> **Updated:** 2026-04-23
> **Focus:** agent design patterns, ReAct, Function Calling, multi-agent collaboration

## 📋 Contents

1. [Fundamental Concepts](#1-fundamental-concepts)
2. [Design Pattern Questions](#2-design-pattern-questions)
3. [Engineering Practice Questions](#3-engineering-practice-questions)
4. [High-Score Answer Templates](#4-high-score-answer-templates)

## 1. Fundamental Concepts

### Q1: What is an AI Agent? What are its core components?

<details>
<summary>💡 Key points</summary>

**AI Agent = AI that can reason, decide, and act.**

**Core components:**
```
┌─────────────────────────────────────────┐
│               AI Agent                  │
├─────────────────────────────────────────┤
│ 1. LLM (brain)      - makes decisions   │
│ 2. Tools (actuators)- executes actions  │
│ 3. Memory           - short-term + long-term |
│ 4. Planning         - task breakdown and reflection |
└─────────────────────────────────────────┘
```

**Simple example:**
- User asks: "Book a flight to Shanghai."
- Agent decides to call `search_flights`, then `book_flight`.
- The agent uses tools to complete the task and returns a real result.

**Interview phrasing:**
> "The difference between an agent and a plain LLM is that the LLM only talks, while the agent can do work. The agent uses tools like search, APIs, and databases to complete actual tasks."

</details>

### Q2: What is the ReAct pattern? What is the complete flow?

<details>
<summary>💡 Key points</summary>

**ReAct = Reasoning + Acting.**

**Full flow:**
```
1. Thought: decide what to do next
2. Action: call a tool or function
3. Observation: receive the tool output
4. Repeat 1-3 until the task is done
5. Final Answer: return the final result
```

**Example interaction:**
```
User: "What is the weather in Tokyo and convert it to Fahrenheit?"
Thought: I should get the weather first, then convert units.
Action: get_weather
Action Input: {"city":"Tokyo","unit":"celsius"}
Observation: {"temp":20}
Thought: Convert Celsius to Fahrenheit.
Action: convert_temperature
Action Input: {"value":20,"from":"celsius","to":"fahrenheit"}
Observation: {"value":68}
Final Answer: "The temperature in Tokyo is 68°F."
```

**Best use:** when the task requires multiple steps and external tools.

</details>

### Q3: What is Function Calling and how does it work?

<details>
<summary>💡 Key points</summary>

**Function Calling lets the LLM invoke external functions.**

**How it works:**
1. Define a function schema: name, parameters, description.
2. Register the tool with the model.
3. The model returns a `function_call` instead of plain text.
4. Execute the function with parsed arguments.
5. Return the result to the model and generate the final answer.

**Simple example:**
- User asks: "Get the latest USD to EUR rate."
- Model returns: `get_exchange_rate({"from":"USD","to":"EUR"})`
- Agent executes the function and returns the rate.

**Interview phrasing:**
> "Function Calling turns unstructured natural language into structured JSON. In practice, I used it to let business users query SQL databases with plain language, which made analytics easier for non-technical people."

</details>

## 2. Design Pattern Questions

### Q4: How do you prevent an agent from entering a loop?

<details>
<summary>💡 Key points</summary>

**Why loops happen:**
1. Tool calls fail and the agent retries repeatedly.
2. The task is too complex for one run.
3. Poor prompt design causes repeated behavior.

**Solution:**
```python
max_iterations = 10
iteration = 0
visited = set()  # record actions already taken

while iteration < max_iterations:
    action = agent.thought()
    if action in visited:
        break  # detect a loop
    visited.add(action)
    result = agent.act(action)
    iteration += 1
```

**Protective measures:**
- limit the round count (for example, 10).
- add a timeout if no progress occurs.
- de-duplicate tool calls by tool name + parameters.
- add a reflection step to evaluate progress.
- allow human intervention for hard cases.

</details>

### Q5: What is the difference between Plan-and-Execute and ReAct?

<details>
<summary>💡 Key points</summary>

| Dimension | ReAct | Plan-and-Execute |
|-----------|-------|------------------|
| Flow | think → act → observe (loop) | plan first, then execute |
| Control | lower, more dynamic | higher, more predictable |
| Explainability | medium | high |
| Best for | exploratory tasks | deterministic tasks |

**Plan-and-Execute flow:**
```
1. Planner: break the task into steps
   ["search weather", "check flights", "book hotel"]
2. Executor: execute each step in order
3. Optionally adjust the plan if a step fails
```

**Example:**
- The planner creates a checklist for booking travel.
- The executor runs each step one by one.
- If a step fails, the plan may update the next steps.

</details>

### Q6: How do you design multi-agent collaboration?

<details>
<summary>💡 Key points</summary>

**Typical architecture:**
```
User request
    |
    v
Coordinator (task router)
   /   |   \   \
Research Writer Reviewer Executor
```

**Simple example:**
- `Researcher`: collects facts and data.
- `Writer`: drafts responses or articles.
- `Reviewer`: checks quality and compliance.
- `Executor`: performs the final action, such as publishing.

**Practical example:**
> "I built a content creation system where the Researcher collected sources, the Writer drafted text, the Reviewer checked for errors, and the Executor published the article. This multi-agent flow improved throughput by 3x."

</details>

## 3. Engineering Practice Questions

### Q7: What kinds of agents have you designed?

<details>
<summary>💡 High-score answer</summary>

**Example 1: Customer support agent**
```
Function: answer common customer questions.
Architecture: intent detection → RAG retrieval → answer generation → human fallback.
Result: handled 80% of routine queries and reduced support costs by 60%.
```

**Example 2: Data analysis agent**
```
Function: let users query data in natural language.
Architecture: NL-to-SQL → SQL execution → visualization.
Result: non-technical users could run reports on their own.
```

**Example 3: Code generation agent**
```
Function: generate code from requirements.
Architecture: requirement understanding → code generation → unit testing → auto-fix.
Result: sped up simple feature delivery by 50%.
```

</details>

### Q8: How do you design agent memory?

<details>
<summary>💡 Key points</summary>

**Short-term memory:**
- store the last N dialogue turns.
- use a list or ring buffer.
- summarize or truncate when full.

**Long-term memory:**
- store important facts in a vector database.
- retrieve related memory when needed.
- support forgetting obsolete information.

**Example implementation:**
```python
class AgentMemory:
    def __init__(self):
        self.short_term = []  # last 10 messages
        self.long_term = VectorStore()

    def add(self, message):
        self.short_term.append(message)
        if len(self.short_term) > 10:
            summary = self.summarize(self.short_term[:5])
            self.long_term.add(summary)
            self.short_term = self.short_term[5:]

    def get(self, query):
        memories = self.long_term.search(query, k=3)
        return memories + self.short_term
```

</details>

## 4. High-Score Answer Templates

### 🌟 Strong points when talking about agents

**Avoid:** "The agent just calls tools."

**Better:**
> "I think the core of an agent is the closed loop. After the model generates an answer, I add a Reviewer step that asks, 'Does this answer meet all user requirements?' If not, it reruns. This reflection mechanism improved reliability by 40%."

### 🌟 Strong points when talking about Function Calling

**Avoid:** "It just calls external APIs."

**Better:**
> "Function Calling transforms unstructured language into structured JSON. In practice, I used it to let users query SQL databases with natural language, making analytics accessible to non-technical people. I also added permission checks and parameter whitelists to prevent unauthorized access."

### Q9: What is LangGraph? How do you build complex agent workflows?

<details>
<summary>💡 Key points</summary>

**LangGraph = building stateful agent applications using a graph structure.**

**Why LangGraph?**

| Scenario | LangChain (chain) | LangGraph (graph) |
|----------|-------------------|-------------------|
| simple dialogue | ✅ good | ❌ overkill |
| loops | ❌ not native | ✅ native support |
| conditional branches | ❌ hard | ✅ easy |
| multi-agent collaboration | ❌ complex | ✅ clean |

**Core concepts:**

### 1. Graph structure
```python
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: list
    next_action: str

workflow = StateGraph(AgentState)
workflow.add_node("researcher", research_node)
workflow.add_node("writer", write_node)
workflow.add_node("reviewer", review_node)
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    should_continue,
    {
        "continue": "writer",
        "end": END
    }
)
```

### 2. State management
```python
def research_node(state: AgentState):
    query = state["messages"][-1]
    results = search_tool(query)
    return {
        "messages": state["messages"] + [results],
        "next_action": "write"
    }


def write_node(state: AgentState):
    research_data = state["messages"][-1]
    draft = llm.generate(f"Write an article based on: {research_data}")
    return {
        "messages": state["messages"] + [draft],
        "next_action": "review"
    }
```

### 3. Loops and branches
```python
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    score = llm.evaluate(last_message)
    if score > 8:
        return "end"
    return "continue"
```

**Full example: writing agent**
```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

class WritingState(TypedDict):
    topic: str
    outline: str
    draft: str
    revision_count: int


def outline_node(state):
    outline = llm.invoke(f"Create an outline for '{state['topic']}'")
    return {"outline": outline.content}


def draft_node(state):
    draft = llm.invoke(f"Write an article based on the outline:\n{state['outline']}")
    return {"draft": draft.content}


def review_node(state):
    review = llm.invoke(f"Rate the draft from 1-10:\n{state['draft']}")
    score = int(review.content)
    return {"revision_count": state.get("revision_count", 0) + 1}


def should_revise(state):
    if state.get("revision_count", 0) >= 3:
        return "end"
    score = llm.invoke(f"Score this draft from 1-10:\n{state['draft']}")
    if int(score.content) >= 8:
        return "end"
    return "revise"

workflow = StateGraph(WritingState)
workflow.add_node("outline", outline_node)
workflow.add_node("draft", draft_node)
workflow.add_node("review", review_node)
workflow.set_entry_point("outline")
workflow.add_edge("outline", "draft")
workflow.add_edge("draft", "review")
workflow.add_conditional_edges(
    "review",
    should_revise,
    {"revise": "draft", "end": END}
)

app = workflow.compile()
result = app.invoke({"topic": "The future of AI agents"})
```

**LangGraph vs AutoGPT:**

| Feature | AutoGPT | LangGraph |
|---------|---------|-----------|
| Control | low | high |
| Reliability | low | high |
| Best use | exploration | production workflows |
| Cost | high | manageable |

**Interview phrasing:**
> "LangGraph solves LangChain's pain points around loops and complex branching. We used it to build a writing agent that iterates from outline → draft → review → revision until quality criteria are met. It's more controllable than AutoGPT and more flexible than pure prompt-based workflows."

</details>

---

### Q10: What is the full tool calling flow? How do you handle failures?

<details>
<summary>💡 Key points</summary>

**Tool calling flow: detect → extract params → execute → handle result.**

### Step 1: define tools
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. Beijing"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search a database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "table": {"type": "string"}
                },
                "required": ["query", "table"]
            }
        }
    }
]
```

### Step 2: model decides
```python
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is the weather in Beijing?"}],
    tools=tools,
    tool_choice="auto"
)
```

Then the model may return:
```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "id": "call_123",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"Beijing\", \"unit\": \"celsius\"}"
      }
    }
  ]
}
```

### Step 3: validate and execute
```python
def execute_tool_call(tool_call):
    function_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    if function_name == "get_weather":
        return get_weather(**args)
    if function_name == "search_database":
        return search_database(**args)
```

### Step 4: handle failure
- validate parameters before execution
- retry once for transient errors
- fallback to a safe answer if the tool fails
- log failures and add a human review step for critical tasks

**Simple failure example:**
- Tool: `get_weather(city="")`
- Detect invalid city and return an error message instead of retrying blindly.

</details>
