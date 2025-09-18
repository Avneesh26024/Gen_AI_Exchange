import asyncio
from Tools.Web_Search import web_search_tool as async_web_search_tool
from Tools.Reverse_Image_Search import reverse_image_search as async_reverse_image_search
from typing import List, Dict, Any

def web_search_tool(query: str) -> List[Dict[str, Any]]:
    """Synchronous wrapper for the async web_search_tool."""
    return asyncio.run(async_web_search_tool(query))

def reverse_image_search(image_path: str) -> dict:
    """Synchronous wrapper for the async reverse_image_search tool."""
    return asyncio.run(async_reverse_image_search.ainvoke({"image_path": image_path}))

