"""
Email Assistant with HITL, Memory, AND Experience Packs.

Extends email_assistant_hitl_memory.py with:
  1. Structured reasoning traces (StepTrace) at triage and response stages
  2. Experience Pack retrieval — injects relevant lessons before triage
  3. Interactive coaching via LangGraph interrupt (not blocking input())
  4. Proper LangGraph state conventions (partial-dict returns, reducers)
"""

from typing import Annotated, Literal, List
import json

from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore
from langgraph.types import interrupt, Command

from email_assistant.tools import get_tools, get_tools_by_name
from email_assistant.tools.default.prompt_templates import HITL_MEMORY_TOOLS_PROMPT
from email_assistant.prompts import (
    triage_system_prompt,
    triage_user_prompt,
    agent_system_prompt_hitl_memory,
    default_triage_instructions,
    default_background,
    default_response_preferences,
    default_cal_preferences,
    MEMORY_UPDATE_INSTRUCTIONS,
    MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT,
)
from email_assistant.schemas import RouterSchema, StateInput, UserPreferences
from email_assistant.utils import parse_email, format_for_display, format_email_markdown, _html_to_text
from email_assistant.experience_packs import (
    StepTrace,
    ExperiencePack,
    ExperienceLibrary,
    trace_reducer,
    COACHING_PACK_PROMPT,
)
from dotenv import load_dotenv

load_dotenv(".env")


# ---------------------------------------------------------------------------
# Extended State  (adds trace + experience_packs to the base State)
# ---------------------------------------------------------------------------

from typing_extensions import TypedDict
from langgraph.graph import MessagesState


class State(MessagesState):
    """Graph state — extends the base MessagesState with Experience Pack fields."""

    email_input: dict
    classification_decision: Literal["ignore", "respond", "notify"]
    # Experience Pack additions
    trace: Annotated[List[StepTrace], trace_reducer]
    experience_packs: List[ExperiencePack]


# ---------------------------------------------------------------------------
# Tools & LLM setup  (same as original)
# ---------------------------------------------------------------------------

tools = get_tools(
    ["write_email", "schedule_meeting", "check_calendar_availability", "Question", "Done"]
)
tools_by_name = get_tools_by_name(tools)

llm = init_chat_model("openai:gpt-4.1", temperature=0.0)
llm_router = llm.with_structured_output(RouterSchema)
llm_with_tools = llm.bind_tools(tools, tool_choice="required")


# ---------------------------------------------------------------------------
# Memory helpers  (copied verbatim from original)
# ---------------------------------------------------------------------------


def get_memory(store, namespace, default_content=None):
    user_preferences = store.get(namespace, "user_preferences")
    if user_preferences:
        return user_preferences.value
    else:
        store.put(namespace, "user_preferences", default_content)
        return default_content


def update_memory(store, namespace, messages):
    user_preferences = store.get(namespace, "user_preferences")
    _llm = init_chat_model("openai:gpt-4.1", temperature=0.0).with_structured_output(
        UserPreferences
    )
    result = _llm.invoke(
        [
            {
                "role": "system",
                "content": MEMORY_UPDATE_INSTRUCTIONS.format(
                    current_profile=user_preferences.value, namespace=namespace
                ),
            },
        ]
        + messages
    )
    store.put(namespace, "user_preferences", result.user_preferences)


# ---------------------------------------------------------------------------
# NEW NODE: Retrieve Experience Packs  (runs before triage)
# ---------------------------------------------------------------------------


def retrieve_experience(state: State, store: BaseStore) -> dict:
    """Pull relevant Experience Packs from the store and inject into state."""
    email_body = _html_to_text(state["email_input"].get("email_thread", ""))
    subject = state["email_input"].get("subject", "")
    query = f"{subject} {email_body}"

    lib = ExperienceLibrary(store)
    relevant = lib.retrieve(query, k=3)

    # Increment usage counts for the packs we're about to use
    if relevant:
        lib.increment_usage([p.pack_id for p in relevant])
        print(f"📦 Injecting {len(relevant)} Experience Pack(s)")

    return {"experience_packs": relevant}


# ---------------------------------------------------------------------------
# MODIFIED NODE: Triage Router  (now emits a StepTrace)
# ---------------------------------------------------------------------------


def triage_router(
    state: State, store: BaseStore
) -> Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]:
    """Triage email with structured trace and experience pack awareness."""

    author, to, subject, email_thread = parse_email(state["email_input"])
    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    triage_instructions = get_memory(
        store, ("email_assistant", "triage_preferences"), default_triage_instructions
    )

    # Build experience pack context
    pack_directives = ""
    injected_packs = state.get("experience_packs", [])
    if injected_packs:
        pack_directives = "\n\n< Experience Packs (lessons from past coaching) >\n"
        for p in injected_packs:
            pack_directives += f"- WHEN: {p.trigger_context}\n  DO: {p.directive}\n"
        pack_directives += "</ Experience Packs >\n"

    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=triage_instructions,
    ) + pack_directives

    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    classification = result.classification

    # Build structured trace for this triage decision
    trace_entry = StepTrace(
        objective="Classify incoming email as ignore/notify/respond",
        instructions_received=f"Triage rules + {len(injected_packs)} experience pack(s)",
        options_considered=["ignore", "notify", "respond"],
        chosen_option=classification,
        rationale=result.reasoning,
        confidence=0.9 if classification in ("ignore", "respond") else 0.7,
    )

    if classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "classification_decision": result.classification,
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email: {email_markdown}",
                }
            ],
            "trace": [trace_entry],
        }
    elif classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        goto = "coaching"
        update = {
            "classification_decision": classification,
            "trace": [trace_entry],
        }
    elif classification == "notify":
        print("🔔 Classification: NOTIFY - This email contains important information")
        goto = "triage_interrupt_handler"
        update = {
            "classification_decision": classification,
            "trace": [trace_entry],
        }
    else:
        raise ValueError(f"Invalid classification: {classification}")

    return Command(goto=goto, update=update)


# ---------------------------------------------------------------------------
# Triage interrupt handler  (same as original, adds trace)
# ---------------------------------------------------------------------------


def triage_interrupt_handler(
    state: State, store: BaseStore
) -> Command[Literal["response_agent", "coaching"]]:
    author, to, subject, email_thread = parse_email(state["email_input"])
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    messages = [
        {
            "role": "user",
            "content": f"Email to notify user about: {email_markdown}",
        }
    ]

    request = {
        "action_request": {
            "action": f"Email Assistant: {state['classification_decision']}",
            "args": {},
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,
            "allow_accept": False,
        },
        "description": email_markdown,
    }

    response = interrupt([request])[0]

    if response["type"] == "response":
        user_input = response["args"]
        messages.append(
            {
                "role": "user",
                "content": f"User wants to reply to the email. Use this feedback to respond: {user_input}",
            }
        )
        update_memory(
            store,
            ("email_assistant", "triage_preferences"),
            [
                {
                    "role": "user",
                    "content": "The user decided to respond to the email, so update the triage preferences to capture this.",
                }
            ]
            + messages,
        )
        goto = "response_agent"

    elif response["type"] == "ignore":
        messages.append(
            {
                "role": "user",
                "content": "The user decided to ignore the email even though it was classified as notify. Update triage preferences to capture this.",
            }
        )
        update_memory(store, ("email_assistant", "triage_preferences"), messages)
        goto = "coaching"

    else:
        raise ValueError(f"Invalid response: {response}")

    return Command(goto=goto, update={"messages": messages})


# ---------------------------------------------------------------------------
# LLM call node  (same as original, injects experience pack context)
# ---------------------------------------------------------------------------


def llm_call(state: State, store: BaseStore) -> dict:
    """LLM decides whether to call a tool or not — with experience pack context."""

    cal_preferences = get_memory(
        store, ("email_assistant", "cal_preferences"), default_cal_preferences
    )
    response_preferences = get_memory(
        store, ("email_assistant", "response_preferences"), default_response_preferences
    )

    # Inject experience pack directives into the system prompt
    pack_context = ""
    injected_packs = state.get("experience_packs", [])
    if injected_packs:
        pack_context = "\n\n< Experience Packs (lessons from past coaching) >\n"
        for p in injected_packs:
            pack_context += f"- WHEN: {p.trigger_context}\n  DO: {p.directive}\n"
        pack_context += "</ Experience Packs >\n"

    system_prompt = agent_system_prompt_hitl_memory.format(
        tools_prompt=HITL_MEMORY_TOOLS_PROMPT,
        background=default_background,
        response_preferences=response_preferences,
        cal_preferences=cal_preferences,
    ) + pack_context

    llm_result = llm_with_tools.invoke(
        [{"role": "system", "content": system_prompt}] + state["messages"]
    )

    # Build trace for this response-agent LLM call
    tool_names = [tc["name"] for tc in (llm_result.tool_calls or [])]
    chosen = tool_names[0] if tool_names else "no tool call"
    trace_entry = StepTrace(
        objective="Draft response or take action on email",
        instructions_received=f"Response prefs + {len(injected_packs)} experience pack(s)",
        options_considered=[t.name for t in tools],
        tools_used=tool_names,
        chosen_option=chosen,
        rationale=llm_result.content[:200] if llm_result.content else f"Called {chosen}",
        confidence=0.8,
    )

    return {
        "messages": [llm_result],
        "trace": [trace_entry],
    }


# ---------------------------------------------------------------------------
# Interrupt handler  (verbatim from original — no changes needed)
# ---------------------------------------------------------------------------


def interrupt_handler(
    state: State, store: BaseStore
) -> Command[Literal["llm_call", "__end__"]]:
    """Creates an interrupt for human review of tool calls."""
    result = []
    goto = "llm_call"

    for tool_call in state["messages"][-1].tool_calls:
        hitl_tools = ["write_email", "schedule_meeting", "Question"]

        if tool_call["name"] not in hitl_tools:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(
                {
                    "role": "tool",
                    "content": observation,
                    "tool_call_id": tool_call["id"],
                }
            )
            continue

        email_input = state["email_input"]
        author, to, subject, email_thread = parse_email(email_input)
        original_email_markdown = format_email_markdown(subject, author, to, email_thread)

        tool_display = format_for_display(tool_call)
        description = original_email_markdown + tool_display

        if tool_call["name"] == "write_email":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True,
            }
        elif tool_call["name"] == "schedule_meeting":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True,
            }
        elif tool_call["name"] == "Question":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": False,
                "allow_accept": False,
            }
        else:
            raise ValueError(f"Invalid tool call: {tool_call['name']}")

        request = {
            "action_request": {
                "action": tool_call["name"],
                "args": tool_call["args"],
            },
            "config": config,
            "description": description,
        }

        response = interrupt([request])[0]

        if response["type"] == "accept":
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(
                {
                    "role": "tool",
                    "content": observation,
                    "tool_call_id": tool_call["id"],
                }
            )

        elif response["type"] == "edit":
            tool = tools_by_name[tool_call["name"]]
            initial_tool_call = tool_call["args"]
            edited_args = response["args"]["args"]

            ai_message = state["messages"][-1]
            current_id = tool_call["id"]
            updated_tool_calls = [
                tc for tc in ai_message.tool_calls if tc["id"] != current_id
            ] + [
                {
                    "type": "tool_call",
                    "name": tool_call["name"],
                    "args": edited_args,
                    "id": current_id,
                }
            ]
            result.append(
                ai_message.model_copy(update={"tool_calls": updated_tool_calls})
            )

            if tool_call["name"] == "write_email":
                observation = tool.invoke(edited_args)
                result.append(
                    {"role": "tool", "content": observation, "tool_call_id": current_id}
                )
                update_memory(
                    store,
                    ("email_assistant", "response_preferences"),
                    [
                        {
                            "role": "user",
                            "content": f"User edited the email response. Here is the initial email generated by the assistant: {initial_tool_call}. Here is the edited email: {edited_args}. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            elif tool_call["name"] == "schedule_meeting":
                observation = tool.invoke(edited_args)
                result.append(
                    {"role": "tool", "content": observation, "tool_call_id": current_id}
                )
                update_memory(
                    store,
                    ("email_assistant", "cal_preferences"),
                    [
                        {
                            "role": "user",
                            "content": f"User edited the calendar invitation. Here is the initial calendar invitation generated by the assistant: {initial_tool_call}. Here is the edited calendar invitation: {edited_args}. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response["type"] == "ignore":
            if tool_call["name"] == "write_email":
                result.append(
                    {
                        "role": "tool",
                        "content": "User ignored this email draft. Ignore this email and end the workflow.",
                        "tool_call_id": tool_call["id"],
                    }
                )
                goto = END
                update_memory(
                    store,
                    ("email_assistant", "triage_preferences"),
                    state["messages"]
                    + result
                    + [
                        {
                            "role": "user",
                            "content": f"The user ignored the email draft. That means they did not want to respond to the email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            elif tool_call["name"] == "schedule_meeting":
                result.append(
                    {
                        "role": "tool",
                        "content": "User ignored this calendar meeting draft. Ignore this email and end the workflow.",
                        "tool_call_id": tool_call["id"],
                    }
                )
                goto = END
                update_memory(
                    store,
                    ("email_assistant", "triage_preferences"),
                    state["messages"]
                    + result
                    + [
                        {
                            "role": "user",
                            "content": f"The user ignored the calendar meeting draft. That means they did not want to schedule a meeting for this email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            elif tool_call["name"] == "Question":
                result.append(
                    {
                        "role": "tool",
                        "content": "User ignored this question. Ignore this email and end the workflow.",
                        "tool_call_id": tool_call["id"],
                    }
                )
                goto = END
                update_memory(
                    store,
                    ("email_assistant", "triage_preferences"),
                    state["messages"]
                    + result
                    + [
                        {
                            "role": "user",
                            "content": f"The user ignored the Question. That means they did not want to answer the question or deal with this email. Update the triage preferences to ensure emails of this type are not classified as respond. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response["type"] == "response":
            user_feedback = response["args"]
            if tool_call["name"] == "write_email":
                result.append(
                    {
                        "role": "tool",
                        "content": f"User gave feedback, which can we incorporate into the email. Feedback: {user_feedback}",
                        "tool_call_id": tool_call["id"],
                    }
                )
                update_memory(
                    store,
                    ("email_assistant", "response_preferences"),
                    state["messages"]
                    + result
                    + [
                        {
                            "role": "user",
                            "content": f"User gave feedback, which we can use to update the response preferences. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            elif tool_call["name"] == "schedule_meeting":
                result.append(
                    {
                        "role": "tool",
                        "content": f"User gave feedback, which can we incorporate into the meeting request. Feedback: {user_feedback}",
                        "tool_call_id": tool_call["id"],
                    }
                )
                update_memory(
                    store,
                    ("email_assistant", "cal_preferences"),
                    state["messages"]
                    + result
                    + [
                        {
                            "role": "user",
                            "content": f"User gave feedback, which we can use to update the calendar preferences. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}.",
                        }
                    ],
                )

            elif tool_call["name"] == "Question":
                result.append(
                    {
                        "role": "tool",
                        "content": f"User answered the question, which can we can use for any follow up actions. Feedback: {user_feedback}",
                        "tool_call_id": tool_call["id"],
                    }
                )
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

    # Build trace for this human review decision
    reviewed_tools = [tc["name"] for tc in state["messages"][-1].tool_calls]
    trace_entry = StepTrace(
        objective=f"Human review of {', '.join(reviewed_tools)}",
        instructions_received="HITL review request",
        options_considered=["accept", "edit", "ignore", "response"],
        tools_used=reviewed_tools,
        chosen_option=response["type"],
        rationale=str(response.get("args", ""))[:200] if response["type"] in ("response", "edit") else f"User chose to {response['type']}",
        confidence=1.0,
    )

    return Command(goto=goto, update={"messages": result, "trace": [trace_entry]})


# ---------------------------------------------------------------------------
# Conditional edge  (same as original)
# ---------------------------------------------------------------------------


def should_continue(state: State, store: BaseStore) -> Literal["interrupt_handler", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "Done":
                return END
            else:
                return "interrupt_handler"


# ---------------------------------------------------------------------------
# NEW NODE: Coaching session  (via LangGraph interrupt, not input())
# ---------------------------------------------------------------------------


def coaching_node(state: State, store: BaseStore) -> dict:
    """Present the trace to the human coach and generate an ExperiencePack.

    Uses LangGraph interrupt so it works with Agent Inbox, not just terminal.
    """
    trace = state.get("trace", [])

    # Format the trace for display
    trace_display = "## Reasoning Trace\n\n"
    for t in trace:
        trace_display += f"**Step {t.step_id}**: {t.objective}\n"
        trace_display += f"- Chosen: {t.chosen_option}\n"
        trace_display += f"- Rationale: {t.rationale}\n"
        trace_display += f"- Confidence: {t.confidence}\n\n"

    # Create interrupt for coaching
    request = {
        "action_request": {
            "action": "Coach the agent",
            "args": {},
        },
        "config": {
            "allow_ignore": True,  # skip coaching
            "allow_respond": True,  # provide feedback
            "allow_edit": False,
            "allow_accept": False,
        },
        "description": trace_display,
    }

    response = interrupt([request])[0]

    if response["type"] == "ignore":
        # User chose not to coach — no pack created
        return {}

    if response["type"] == "response":
        feedback = response["args"]

        # Serialize trace for the LLM
        trace_json = json.dumps(
            [t.model_dump() for t in trace[-3:]], indent=2
        )
        email_json = json.dumps(state["email_input"], indent=2)

        prompt = COACHING_PACK_PROMPT.format(
            trace_json=trace_json,
            email_json=email_json,
            feedback=feedback,
        )

        pack_llm = init_chat_model("openai:gpt-4.1", temperature=0.0).with_structured_output(
            ExperiencePack
        )
        new_pack: ExperiencePack = pack_llm.invoke(prompt)

        # Persist through the store
        lib = ExperienceLibrary(store)
        lib.add_pack(new_pack)

        return {}

    return {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

# Inner response agent (same structure as original)
agent_builder = StateGraph(State)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("interrupt_handler", interrupt_handler)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {"interrupt_handler": "interrupt_handler", END: END},
)
response_agent = agent_builder.compile()


# Outer workflow with experience packs
overall_workflow = StateGraph(State, input=StateInput)
overall_workflow.add_node("retrieve_experience", retrieve_experience)
overall_workflow.add_node("triage_router", triage_router)
overall_workflow.add_node("triage_interrupt_handler", triage_interrupt_handler)
overall_workflow.add_node("response_agent", response_agent)
overall_workflow.add_node("coaching", coaching_node)

# Edges
overall_workflow.add_edge(START, "retrieve_experience")
overall_workflow.add_edge("retrieve_experience", "triage_router")
# triage_router uses Command() to route to triage_interrupt_handler, response_agent, or coaching
# triage_interrupt_handler uses Command() to route to response_agent or coaching
# response_agent flows to coaching
overall_workflow.add_edge("response_agent", "coaching")
overall_workflow.add_edge("coaching", END)

# Compile
email_assistant = overall_workflow.compile()
