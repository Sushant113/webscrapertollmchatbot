import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def scrape_website(url, depth=5):
    to_visit = [(url, 0)]
    visited = set()
    data = []

    while to_visit:
        current_url, current_depth = to_visit.pop(0)
        if current_depth > depth or current_url in visited:
            continue
        
        try:
            response = requests.get(current_url)
            if response.status_code != 200:
                continue
        except requests.RequestException:
            continue

        visited.add(current_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        data.append((current_url, soup.get_text()))

        if current_depth < depth:
            for link in soup.find_all('a', href=True):
                full_url = urljoin(current_url, link['href'])
                if full_url.startswith(url):  # Only follow links within the same domain
                    to_visit.append((full_url, current_depth + 1))

    return data
