from langgraph.tools import tool


@tool
def human_response(prompt: str) -> str:
    """
    Tool for getting additional input from the human user.
    The agent calls this when it needs more info.

    Args:
        prompt (str): The question or context for the user.

    Returns:
        str: The user's input.
    """
    response = input(f"{prompt}\nYour response: ")
    return response
