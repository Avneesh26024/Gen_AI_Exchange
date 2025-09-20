# In your Tools/Scraper.py file

import asyncio
import random
from playwright.async_api import async_playwright
import trafilatura

# ✅ A list of modern, realistic User-Agents to rotate through
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0",
]

async def fetch_and_extract_content(url: str, min_text_len: int = 200):
    """
    Launches a stealthier browser and uses Trafilatura to extract the main article content.
    """
    browser = None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            
            # ✅ Select a random User-Agent for this specific request
            page = await browser.new_page(user_agent=random.choice(USER_AGENTS))
            
            # ✅ Use a longer timeout and wait for 'networkidle' - this is much more robust
            # It waits until there are no new network connections for 500 ms.
            await page.goto(url, wait_until="networkidle", timeout=45000)

            html_content = await page.content()
            await browser.close()

            # Trafilatura logic remains the same - it's already doing a great job
            main_text = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=False,
                no_fallback=True
            )

            if main_text and len(main_text) >= min_text_len:
                print(f"✅ Successfully scraped: {url}")
                return True, main_text
            else:
                # If Trafilatura found nothing, the page was likely a block/CAPTCHA page
                print(f"❌ Failed (No main content found by Trafilatura): {url}")
                return False, None

    except Exception as e:
        if browser:
            await browser.close()
        print(f"❌ Scraping error for {url}: {e}")
        return False, None