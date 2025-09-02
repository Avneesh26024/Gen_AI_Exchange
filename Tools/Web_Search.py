import os
from dotenv import load_dotenv
import requests
import asyncio
from Tools.Scraper import fetch_and_extract_content
from langchain_core.tools import tool
from typing import List, Dict, Any
load_dotenv()

# This function remains unchanged as requested.
def google_search(query, api_key=os.getenv("GOOGLE_API_KEY"), cse_id=os.getenv("CSE_ID"), num_results=5):
    """
    Perform a Google Custom Search.
    query: str -> Search query
    api_key: str -> Google API Key
    cse_id: str -> Custom Search Engine ID
    num_results: int -> Number of results to fetch (max 10 per request)
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": api_key, "cx": cse_id, "num": num_results}
    response = requests.get(url, params=params)
    results = response.json()
    search_items = []
    if "items" in results:
        for item in results["items"]:
            search_items.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            })
    return search_items


# --- The LangGraph Tool ---

async def web_search_tool(query: str) -> List[Dict[str, Any]]:
    """
    Searches Google for a query, scrapes the content of the top results concurrently,
    and returns a structured list of {title, url, content}.
    """
    print(f"🔎 Searching the web for: '{query}'")
    search_results = google_search(query)

    if not search_results:
        return []

    # Create tasks to scrape each URL concurrently
    scraping_tasks = [fetch_and_extract_content(item['link'], min_text_len=400) for item in search_results]
    scraped_data = await asyncio.gather(*scraping_tasks)

    results = []
    for item, (is_scrapable, content) in zip(search_results, scraped_data):
        results.append({
            "title": item['title'],
            "url": item['link'],
            "content": content if is_scrapable else None,
            "scrapable": is_scrapable
        })

    return results

async def main():
    query = "Latest AI research 2025"
    result = await web_search_tool(query)
    print("\n📄 Final Report:\n")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
