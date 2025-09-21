from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
import json

# --- 1. Import your node functions from the new 'nodes.py' file ---
from nodes import (
    router as verifier_router,
    add_existing_evidence,
    text_evidence_collection,
    image_analysis,
    verify_claims,
    save_results_to_cache,
    llm,
    summarizer_llm,
    validation_llm
)
from Retriever.Retriever import retrieve_chunks
from prompts import checker_prompt, final_answer_prompt


# --- 2. Define a single, UNIFIED state for the entire application ---
class FullAgentState(TypedDict):
    # Conversational state
    chat_history: List[str]
    images: List[str]
    retrieved_evidence: List[dict]

    # Verification sub-task state (used by nodes from nodes.py)
    text_news: List[str]
    image_path: List[str]  # Added for compatibility
    unverified_claims: List[str]
    already_verified: List[Dict]
    text_with_evidence: str
    image_analysis: str
    save_to_vector_db: bool
    verified_results: Dict


# --- 3. Define all nodes and decision functions ---

def intent_checker_node(state: FullAgentState) -> FullAgentState:
    """A simple entry node that just passes the state through."""
    print("-> Entering intent checker node...")
    return state


def decide_intent(state: FullAgentState) -> str:
    """Checks the user's intent to route the conversation."""
    print("-> 🗣️ Intent Checker: Analyzing user query...")
    if state.get("images"):
        return "VERIFY"
    messages = state.get("chat_history", [])
    if len(messages) < 2:
        return "VERIFY"

    user_query = messages[-1]
    chat_context = json.dumps(messages[:-1])
    response = llm.invoke(checker_prompt(chat_context, user_query))
    intent = response.content.strip().upper()
    print(f"-> Intent Classified as: {intent}")
    return intent if intent in ["VERIFY", "RETRIEVE_EVIDENCE", "FINAL_ANSWER", "AMBIGUOUS"] else "VERIFY"


def prepare_for_verification(state: FullAgentState) -> FullAgentState:
    """Prepares the state to be passed into the verification sub-flow."""
    print("-> Preparing state for verification sub-graph...")
    user_query = state["chat_history"][-1]
    # Set the keys that the verifier nodes expect
    state["text_news"] = [user_query]
    state["image_path"] = state.get("images", [])  # CRITICAL FIX: Maps 'images' to 'image_path'
    state["save_to_vector_db"] = True
    # Clear previous run's data
    state["text_with_evidence"] = ""
    state["image_analysis"] = ""
    state["verified_results"] = {}
    return state


def _get_last_claim_from_history(chat_history: List[str]) -> str:
    """Finds the last user query that likely triggered a verification."""
    if len(chat_history) >= 2:
        return chat_history[-2]
    return None


def retriever_node(state: FullAgentState) -> FullAgentState:
    """Retrieves evidence for the last verified claim."""
    print("-> 📚 Retrieving evidence from Vector DB...")
    last_claim = _get_last_claim_from_history(state["chat_history"])
    if not last_claim:
        state["retrieved_evidence"] = []
        return state
    state["retrieved_evidence"] = retrieve_chunks(last_claim, top_k=5)
    return state


def decide_after_retrieval(state: FullAgentState) -> str:
    """Decides where to go after attempting to retrieve evidence."""
    return "ASK_CLARIFICATION" if not state.get("retrieved_evidence") else "PROCEED_TO_ANSWER"


def clarification_node(state: FullAgentState) -> FullAgentState:
    """Handles ambiguous cases by asking the user for more information."""
    print("-> Asking for clarification...")
    message = "I'm not sure how to help. Could you rephrase your request as a claim to be verified or ask for evidence on a previous claim?"
    state["chat_history"].append(message)
    return state


def final_answer_node(state: FullAgentState) -> FullAgentState:
    """Generates the final, conversational response for the user."""
    print("-> ✍️ Generating final answer...")

    if state.get("verified_results"):  # Case: A verification just finished
        results = state["verified_results"]
        new_results = results.get("text_claims", [])
        if new_results:
            decision = new_results[0].get("final_decision", "Verification complete.")
            summary = f"Regarding the claim '{new_results[0]['claim']}', my finding is: {decision}"
        else:
            cached = state.get("already_verified", [])
            if cached:
                summary = f"Regarding the claim '{cached[0]['claim']}', I found this in my knowledge base: {cached[0]['result']}"
            else:
                summary = "Verification process completed, but no specific decision was found."
        state["chat_history"].append(summary)
    else:  # Case: Answering based on retrieved evidence or simple history
        prompt = final_answer_prompt(
            state.get("chat_history", []),
            state.get("retrieved_evidence", [])
        )
        response = llm.invoke(prompt)
        state["chat_history"].append(response.content)

    state["retrieved_evidence"] = []  # Clear evidence after use
    return state


# --- 4. Define the single, unified graph ---
workflow = StateGraph(FullAgentState)

# Add ALL nodes to this single graph
workflow.add_node("intent_checker", intent_checker_node)
workflow.add_node("prepare_for_verification", prepare_for_verification)
workflow.add_node("retrieve_evidence", retriever_node)
workflow.add_node("clarification", clarification_node)
workflow.add_node("final_answer", final_answer_node)
workflow.add_node("verifier_router", verifier_router)
workflow.add_node("add_existing_evidence", add_existing_evidence)
workflow.add_node("text_evidence_collection", text_evidence_collection)
workflow.add_node("image_analysis", image_analysis)
workflow.add_node("verify_claims", verify_claims)
workflow.add_node("save_cache", save_results_to_cache)

# --- Define the graph's flow ---
workflow.set_entry_point("intent_checker")

workflow.add_conditional_edges("intent_checker", decide_intent, {
    "VERIFY": "prepare_for_verification",
    "RETRIEVE_EVIDENCE": "retrieve_evidence",
    "FINAL_ANSWER": "final_answer",
    "AMBIGUOUS": "clarification",
})

workflow.add_edge("prepare_for_verification", "verifier_router")
workflow.add_edge("verifier_router", "add_existing_evidence")
workflow.add_edge("add_existing_evidence", "text_evidence_collection")
workflow.add_edge("text_evidence_collection", "image_analysis")
workflow.add_edge("image_analysis", "verify_claims")
workflow.add_edge("verify_claims", "save_cache")
workflow.add_edge("save_cache", "final_answer")

workflow.add_conditional_edges("retrieve_evidence", decide_after_retrieval, {
    "ASK_CLARIFICATION": "clarification",
    "PROCEED_TO_ANSWER": "final_answer",
})

workflow.add_edge("clarification", END)
workflow.add_edge("final_answer", END)

# Compile the final agent
main_agent = workflow.compile()
class ConversationManager:
    """Manages the conversation state."""
    def __init__(self):
        self.chat_history = []

    def run_agent(self, query: str, images: List[str] = None):
        """Runs the agent for a single turn of the conversation."""
        if images is None:
            images = []

        # Add the new user query to the history
        self.chat_history.append(query)

        initial_state = {
            "chat_history": self.chat_history,
            "images": images,
            "retrieved_evidence": [] # Initialize with empty evidence
        }

        # Invoke the agent
        final_state = main_agent.invoke(initial_state)

        # Update the history with the final state from the agent
        self.chat_history = final_state.get("chat_history", [])

        print("\n--- Agent Response ---")
        # Print the last message from the agent
        print(self.chat_history[-1])

# Example usage:
if __name__ == "__main__":
    manager = ConversationManager()

    # Example 1: A query that needs verification
    print("--- Running Example 1: New Claim ---")
    manager.run_agent("Is it true that the earth is flat?")

    # Example 2: A follow-up query asking for evidence
    print("\n--- Running Example 2: Evidence Request ---")
    manager.run_agent("How do you know that? Show me the proof.")

    # Example 3: A simple follow-up that can be answered from history
    print("\n--- Running Example 3: Simple Follow-up ---")
    manager.run_agent("What was the final decision on that claim?")

    # Example 4: An ambiguous request that should trigger clarification
    print("\n--- Running Example 4: Ambiguous Request ---")
    # Create a new manager for a clean history for this test
    ambiguous_manager = ConversationManager()
    ambiguous_manager.run_agent("What about the other thing?")
