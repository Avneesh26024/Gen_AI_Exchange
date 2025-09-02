import asyncio
from playwright.async_api import async_playwright
import trafilatura

async def fetch_and_extract_content(url: str, min_text_len: int = 200):
    """
    Launches a browser, navigates to the URL, and uses Trafilatura to extract the main article content.
    Returns a tuple: (is_scrapable, content)
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            await page.wait_for_timeout(3000)  # Let the page settle

            html_content = await page.content()
            await browser.close()

            main_text = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=False,
                no_fallback=True
            )

            if main_text and len(main_text) >= min_text_len:

                return True, main_text
            else:
                return False, f"Content too short or not found ({len(main_text) if main_text else 0} chars)."

    except Exception as e:
        return False, f"An error occurred: {e}"


