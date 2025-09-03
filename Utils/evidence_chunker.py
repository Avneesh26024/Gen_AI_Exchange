import re
from typing import List


def chunk_text_by_paragraph(
        raw_text: str,
        min_chunk_size: int = 50
) -> List[str]:
    """
    Splits a raw text into meaningful chunks based on paragraphs.

    Args:
        raw_text (str): The full text content to be chunked.
        min_chunk_size (int): The minimum number of characters for a paragraph to be considered a chunk.

    Returns:
        List[str]: A list of text chunks (strings).
    """
    # Split the text by one or more newline characters to identify paragraphs
    paragraphs = re.split(r'\n\s*\n', raw_text)

    text_chunks = []

    for p in paragraphs:
        # Clean up whitespace from the paragraph
        cleaned_p = p.strip()

        # Only include paragraphs that meet the minimum size requirement
        if len(cleaned_p) >= min_chunk_size:
            text_chunks.append(cleaned_p)

    return text_chunks
