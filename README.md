# Experience Packs: A Coachable Agent

> **Disclaimer:** This project is a learning prototype, not production software. It builds on LangChain's open source [agents-from-scratch](https://github.com/langchain-ai/agents-from-scratch) tutorial and was developed with significant help from AI coding tools, including GitHub Copilot. Its purpose is to show the ideas behind structured reasoning traces, coaching, and reusable experience packs. It is not intended to run a real email workflow at scale.

## Demo Video

Watch the short walkthrough here:

[![Experience Packs demo](https://img.youtube.com/vi/H5TPSXyChQ8/hqdefault.jpg)](https://www.youtube.com/watch?v=H5TPSXyChQ8)

## What This Is

Most AI agents are static. They start with a prompt and some tools, but they do not improve unless a developer changes the code or rewrites the instructions. Experience Packs explores a different idea: an agent that can be coached like a human colleague.

When the agent makes a decision, it can record a structured trace of that decision. A human coach can review the trace, explain what should happen next time, and turn that feedback into an Experience Pack. The next time the agent sees a similar situation, it can retrieve that pack and apply the lesson.

This repository uses an email assistant as the demo surface, but the idea is broader than email. The email workflow is just the easiest way to show the pattern in a concrete form.

## What Is Implemented Right Now

The current repo has three working pieces.

1. **Structured reasoning traces across multiple steps**
   The graph records a `StepTrace` for the triage decision, each response-agent LLM call, and each human review action. Traces capture the objective, options considered, chosen option, rationale, tools used, and a confidence score.

2. **Coaching flow for all outcomes**
   Every terminal outcome (ignore, notify, respond) routes through the coaching node. A coach can review the full multi-step trace and provide feedback. In the graph, this uses LangGraph interrupt support. In the Gmail test script, this also works through terminal input for local testing.

3. **Experience Pack retrieval and reuse**
   Coaching feedback is turned into an `ExperiencePack` with structured fields including applicability criteria, negative examples, and source trace IDs. Packs are stored individually in the LangGraph Store for granular retrieval. In the Gmail test script, packs are also saved to `experience_packs.json` so they can be reused across script runs.

4. **Trace persistence**
   Run traces are appended to `trace_history.jsonl` after each email, so the full decision history is available for later analysis.

## What Is Not Implemented Yet

There are a few gaps between the concepts and the current code.

1. Retrieval uses keyword overlap, not semantic vector search.
2. Confidence values are heuristic, not model-generated.
3. Email sending and calendar actions are still mock implementations.
4. The repo currently assumes a single user context.

## Graph Flow

```text
START -> retrieve_experience -> triage_router -> [coaching | response_agent | triage_interrupt_handler]
                                              triage_interrupt_handler -> [response_agent | coaching]
                                              response_agent -> coaching -> END
```

All terminal outcomes route through coaching so the coach can review and provide feedback on any decision.

## Built On

This project extends [langchain-ai/agents-from-scratch](https://github.com/langchain-ai/agents-from-scratch). The original repository provides the email triage workflow, the human review pattern, tool calling, long term memory through LangGraph Store, and the evaluation dataset. This repo keeps that base structure and adds the Experience Pack layer on top.

## Setup

### Requirements

- Python 3.11 or newer
- OpenAI API key for GPT 4.1
- LangSmith API key if you want tracing in LangSmith

### Install

```shell
git clone <this-repo>
cd coachable-agent

# Create .env with your keys
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=experience-packs
EOF

# Install dependencies
pip install -e ".[dev]"
```

### Run Unit Tests

```shell
python tests/test_experience_packs.py
```

This runs the unit tests for the core models, the Experience Pack library, deduplication, keyword retrieval, and usage tracking.

### Run the Graph Locally

```shell
langgraph dev
```

This starts the `email_assistant_experience` graph locally. You can connect [Agent Inbox](https://github.com/langchain-ai/agent-inbox) to `http://127.0.0.1:2024` and use the graph id `email_assistant_experience`.

## Test with Real Gmail Messages

You can test the full loop against your real inbox. The current Gmail script reads your mail, sends it through the graph, lets you coach the result, and saves packs for reuse. It does **not** send real emails.

### Step 1: Create a Google Cloud project and enable Gmail API

1. Open [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Open **APIs & Services**
4. Open **Library**
5. Search for **Gmail API**
6. Click **Enable**

### Step 2: Configure the OAuth consent screen

1. Open **APIs & Services**
2. Open **OAuth consent screen**
3. Choose **External**
4. Fill in the app name, support email, and developer email
5. Continue through the remaining screens
6. Add your Gmail address as a **Test user**

If you skip the test user step, Google will block the OAuth flow.

### Step 3: Create OAuth credentials

1. Open **APIs & Services**
2. Open **Credentials**
3. Click **Create Credentials**
4. Choose **OAuth client ID**
5. Choose **Desktop app**
6. Create the credential
7. Download the JSON file

### Step 4: Save the credentials file

```shell
mkdir .secrets
cp ~/Downloads/client_secret_*.json .secrets/credentials.json
```

### Step 5: Set your OpenAI API key

Make sure `.env` contains:

```text
OPENAI_API_KEY=sk-your-real-key-here
```

### Step 6: Run the OAuth setup script

```shell
python scripts/setup_gmail.py
```

This opens a browser window so you can sign in and approve Gmail access. The script requests Gmail read access only. When it succeeds, it saves a token file to `.secrets/token.json`.

### Step 7: Process real emails

```shell
# Process up to 3 unread messages from the last 24 hours
python scripts/test_gmail_live.py --minutes 1440 --max-emails 3

# Process messages and coach after each one
python scripts/test_gmail_live.py --minutes 1440 --max-emails 3 --coach

# Filter to a specific recipient address
python scripts/test_gmail_live.py --minutes 1440 --email you@gmail.com --coach
```

With `--coach`, the script prompts you after each email:

```text
Coach feedback (or press Enter to skip):
```

If you give feedback, the script generates an Experience Pack and writes the updated pack library to `experience_packs.json` at the repo root.

## What Is Real and What Is Mock

| Component | Status |
|---|---|
| Gmail message reading | Real |
| Triage classification | Real |
| Triage trace recording | Real |
| Response-agent trace recording | Real |
| Human review trace recording | Real |
| Trace persistence to disk | Real |
| Coaching to Experience Pack | Real |
| Experience Pack retrieval | Real |
| Email sending | Mock |
| Calendar availability | Mock |

## Persistence Model

There are two kinds of persistence in the current repo.

### Experience Packs

Each Experience Pack is stored individually in the LangGraph Store under the `email_assistant / experience_packs` namespace, keyed by its `pack_id`. This allows granular semantic retrieval when the store supports it. In the Gmail test script, packs are also synced to `experience_packs.json` so they survive across separate script runs.

Existing packs stored as a single blob are automatically migrated to individual storage on first load.

### Structured Traces

Structured traces live in graph state during execution and are also appended to `trace_history.jsonl` after each email in the Gmail test script. Each record includes the run ID, timestamp, email metadata, and the full list of step traces.

## Project Structure

```text
src/email_assistant/
├── experience_packs.py           # StepTrace, ExperiencePack, ExperienceLibrary
├── email_assistant_experience.py # Main graph with retrieval, triage, response, coaching
├── schemas.py                    # Base state definitions
├── prompts.py                    # Prompt templates
├── utils.py                      # Formatting and parsing helpers
├── configuration.py              # LangGraph configuration
├── tools/
│   └── default/                  # Mock email and calendar tools
└── eval/
    └── email_dataset.py          # Synthetic email dataset

scripts/
├── setup_gmail.py                # Gmail OAuth setup
└── test_gmail_live.py            # Gmail test script

tests/
└── test_experience_packs.py      # Unit tests

langgraph.json                    # Graph registration
experience_packs.json             # Saved pack library from Gmail test runs
trace_history.jsonl               # Persisted run traces from Gmail test runs
```

## How It Works

### 1. Retrieve Experience

The `retrieve_experience` node looks up relevant Experience Packs for the incoming email and injects them into state.

### 2. Triage with Trace

The `triage_router` node classifies the email as `ignore`, `notify`, or `respond`. The graph records a `StepTrace` for the triage decision.

### 3. Response Agent with Trace

The response agent handles drafting, meeting scheduling, and follow up questions with human review support. Experience Pack directives are injected into the response prompt. Each LLM call and each human review action produce their own `StepTrace` records, capturing tools used, chosen actions, and rationale.

### 4. Coaching

All terminal outcomes route through the `coaching` node. The coach can review the full multi-step trace and provide feedback. If the coach gives feedback, the model converts it into an `ExperiencePack` with structured fields: trigger context, directive, rationale, applicability criteria, negative examples, and source trace IDs.

## Current Limits

1. Retrieval uses keyword overlap, not semantic vector search.
2. Email sending and calendar actions are still mock implementations.
3. Pack deduplication only catches exact trigger matches.
4. Confidence values are heuristic.
5. The repo currently assumes a single user context.

## Credits

1. Base tutorial and implementation surface: [LangChain / agents-from-scratch](https://github.com/langchain-ai/agents-from-scratch)
2. Built with significant assistance from AI coding tools, including GitHub Copilot
3. Experience Packs concept: independent idea, implemented here as a prototype

