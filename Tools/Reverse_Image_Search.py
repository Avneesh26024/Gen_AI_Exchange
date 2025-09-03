from langchain.tools import tool
from google.cloud import vision
import io
import os
from Tools.Scraper import fetch_and_extract_content
import asyncio
# Set your service account key


def reverse_image_search_urls(image_path: str) -> dict:
    """
    Performs reverse image search using Google Cloud Vision API.
    Returns all URLs categorized as:
    - pages_with_matching_images
    - full_matching_images
    - partial_matching_images
    """
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, "rb") as img_file:
        content = img_file.read()

    image = vision.Image(content=content)
    response = client.web_detection(image=image)
    annotations = response.web_detection

    # Helper function to get URLs from items
    def get_urls(items, key='url'):
        if not items:
            return []
        return [getattr(item, key) for item in items]

    results = {
        "pages_with_matching_images": get_urls(annotations.pages_with_matching_images),
        "full_matching_images": get_urls(annotations.full_matching_images),
        "partial_matching_images": get_urls(annotations.partial_matching_images)
    }

    return results


async def scrape_urls_until_target(urls: list[str], target_count: int, category_name: str) -> list[tuple[str, str]]:
    """
    Scrapes URLs from a list until a target number of successful scrapes is met.

    It iterates through the provided list of URLs. If a scrape is successful,
    it adds the result. It stops when the target is reached or the list ends.
    """
    successful_scrapes = []
    url_index = 0

    print(f"-> Starting scrape for category '{category_name}' with target {target_count}.")

    while len(successful_scrapes) < target_count and url_index < len(urls):
        url = urls[url_index]
        success, content = await fetch_and_extract_content(url)
        if success:
            successful_scrapes.append((url, content))
        url_index += 1

    print(f"-> Finished scrape for '{category_name}': Got {len(successful_scrapes)}/{target_count} successful scrapes.")
    return successful_scrapes


# --- Updated Main Tool ---
@tool
async def reverse_image_search(image_path: str) -> dict:
    """
    A comprehensive reverse image search tool with dynamic scraping.
    1. Finds related URLs using Google Vision API.
    2. Scrapes content from URLs in each category until a target number of successes is met.
    3. Returns a dictionary with URLs and their scraped content, grouped by category.
    """
    print("--- Starting Reverse Image Search and Scrape Tool ---")

    # Step 1: Get URLs from the image
    url_data = reverse_image_search_urls(image_path)

    # Step 2: Define targets and create concurrent scraping tasks for each category
    targets = {
        "pages_with_matching_images": 3,
        "full_matching_images": 3,
        "partial_matching_images": 2,
    }

    scraping_tasks = []
    for category, target in targets.items():
        urls = url_data.get(category, [])
        if urls:
            task = scrape_urls_until_target(urls, target, category)
            scraping_tasks.append(task)

    if not scraping_tasks:
        print("--- No URLs found to scrape. ---")
        return {"status": "error", "message": "Could not find any web pages related to the image."}

    # Step 3: Run all category scrapers concurrently
    results_from_all_categories = await asyncio.gather(*scraping_tasks)

    # Step 4: Flatten and structure results into dictionary
    final_results = {"status": "success", "scraped_results": []}
    all_successful_scrapes = [item for sublist in results_from_all_categories for item in sublist]

    for url, content in all_successful_scrapes:
        final_results["scraped_results"].append({
            "url": url,
            "content": content
        })

    print(f"--- Tool Finished: Successfully scraped a total of {len(all_successful_scrapes)} pages. ---")
    return final_results

# --- Example Usage ---
async def main():
    """Main function to run the tool and print the results."""
    image_path = "C:/Users/Avneesh/Desktop/test.png"

    scraped_content = await reverse_image_search.ainvoke({"image_path": image_path})

    print("\n\n==================== FINAL SCRAPED CONTEXT ====================\n")
    print(scraped_content)
    print("==============================================================")


if __name__ == "__main__":
    asyncio.run(main())

