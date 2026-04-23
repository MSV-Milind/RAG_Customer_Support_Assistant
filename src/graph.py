import logging
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from src.config import (
    GROQ_API_KEY,
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TEMP,
    STATE_CONTEXT,
    STATE_ESCALATE,
    STATE_QUERY,
    STATE_RESPONSE,
    SIMILARITY_THRESHOLD,
)
from src.retriever import retrieve
from src.utils import (
    HITL_RESPONSE,
    SYSTEM_PROMPT,
    build_user_prompt,
    format_context,
    log_escalation,
    parse_llm_output,
)

logger = logging.getLogger(__name__)

# State Definition 
class GraphState(TypedDict):
    query:    str
    context:  str
    response: str
    escalate: bool


# Groq LLM 
def _build_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=GROQ_TEMP,
        max_tokens=GROQ_MAX_TOKENS,
        api_key=GROQ_API_KEY,
        stop_sequences=None
    )


# Node 1: Processing Node 
def processing_node(state: GraphState) -> GraphState:
    """
    Core reasoning node.

    1. Checks whether context is non-empty (fallback guard).
    2. Calls Groq LLM with the system prompt + formatted user message.
    3. Parses the response for FINAL ANSWER / ESCALATE.
    4. Updates state accordingly.
    """
    query   = state[STATE_QUERY]
    context = state[STATE_CONTEXT]

    # Fallback
    if not context.strip():
        logger.info("No context available — returning fallback without LLM call.")
        return {
            **state,
            STATE_RESPONSE: "No relevant information found in the knowledge base.",
            STATE_ESCALATE: False,
        }

    # Call Groq LLM 
    llm = _build_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(query, context)),
    ]

    logger.info("Calling Groq LLM (%s) …", GROQ_MODEL)
    response_msg = llm.invoke(messages)
    raw_output   = response_msg.content

    # Parse output 
    escalate, answer = parse_llm_output(raw_output)

    return {
        **state,
        STATE_RESPONSE: answer,
        STATE_ESCALATE: escalate,
    }


# Conditional Router
def route_after_processing(state: GraphState) -> str:
    """
    Routing function called after processing_node.
    Returns the name of the next node to execute.

    If escalate=True: inject HITL message into state and proceed to output.
    LangGraph requires the routing function to return a node name string;
    the HITL mutation happens inside this function before returning.
    """
    if state[STATE_ESCALATE]:
        logger.info("Escalation triggered — routing to HITL path.") 
        state[STATE_RESPONSE] = HITL_RESPONSE
        log_escalation(state)
        return "output_node"

    logger.info("No escalation — routing to output_node.")
    return "output_node"


# Node 2: Output Node
def output_node(state: GraphState) -> GraphState:
    """
    Terminal node. Receives the final state and returns it unchanged.
    In a real deployment this would push the response to a UI / API layer.
    """
    logger.info(
        "Output node reached. escalate=%s  response_len=%d",
        state[STATE_ESCALATE],
        len(state[STATE_RESPONSE]),
    )
    return state


# Graph Builder 
def build_graph():
    """
    Compile and return the LangGraph StateGraph.

    Topology
    --------
    START → processing_node → [conditional] → output_node → END
    """
    builder = StateGraph(GraphState)

    # Register nodes
    builder.add_node("processing_node", processing_node)
    builder.add_node("output_node",     output_node)

    # Edges
    builder.add_edge(START, "processing_node")
    builder.add_conditional_edges(
        "processing_node",
        route_after_processing,
        {
            "output_node": "output_node",
        },
    )
    builder.add_edge("output_node", END)

    return builder.compile()


# High-Level Run Helper
def run_query(query: str, vectorstore, graph=None) -> dict:
    if graph is None:
        graph = build_graph()

    # Retrieve
    chunks  = retrieve(query, vectorstore)
    context = format_context(chunks)

    # Initial state
    initial_state: GraphState = {
        STATE_QUERY:    query,
        STATE_CONTEXT:  context,
        STATE_RESPONSE: "",
        STATE_ESCALATE: False,
    }

    # Graph execution
    final_state = graph.invoke(initial_state)
    return final_state
