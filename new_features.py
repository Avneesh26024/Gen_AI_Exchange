import requests
import xml.etree.ElementTree as ET
from urllib.parse import urljoin

def fetch_server_metadata(url: str):
    data = {
        "url": url,
        "http_headers": {},
        "last_modified": None,
        "etag": None,
        "sitemap_lastmod": [],
        "robots_txt": None
    }

    try:
        # 1. Fetch main URL headers
        response = requests.head(url, allow_redirects=True, timeout=10)
        headers = dict(response.headers)

        data["http_headers"] = headers
        data["last_modified"] = headers.get("Last-Modified")
        data["etag"] = headers.get("ETag")

        # 2. Try to fetch robots.txt
        robots_url = urljoin(url, "/robots.txt")
        try:
            r = requests.get(robots_url, timeout=5)
            if r.status_code == 200:
                data["robots_txt"] = r.text[:500]  # limit preview
        except:
            pass

        # 3. Try to fetch sitemap.xml and parse <lastmod>
        sitemap_url = urljoin(url, "/sitemap.xml")
        try:
            r = requests.get(sitemap_url, timeout=5)
            if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("application/xml"):
                root = ET.fromstring(r.content)
                for elem in root.findall(".//{*}lastmod"):
                    data["sitemap_lastmod"].append(elem.text)
        except:
            pass

    except Exception as e:
        data["error"] = str(e)

    return data


# Example usage
if __name__ == "__main__":
    test_url = "https://www.aljazeera.com/news/2025/9/13/indias-modi-visits-manipur-state-two-years-after-ethnic-clashes"
    result = fetch_server_metadata(test_url)
    import json
    print(json.dumps(result, indent=2))
