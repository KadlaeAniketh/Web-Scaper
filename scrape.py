import requests
from bs4 import BeautifulSoup

def scrape_website(website):
    print("Fetching website using requests...")

    try:
        response = requests.get(website, timeout=10)
        response.raise_for_status()
        html = response.text
        print("Page fetched successfully.")
        return html
    except requests.exceptions.RequestException as e:
        print(f"Error fetching website: {e}")
        return ""

def extract_body_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body
    return str(body_content) if body_content else ""

def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, "html.parser")

    # Remove scripts and styles
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()

    # Clean up text
    cleaned_content = soup.get_text(separator="\n")
    cleaned_content = "\n".join(
        line.strip() for line in cleaned_content.splitlines() if line.strip()
    )

    return cleaned_content

def split_dom_content(dom_content, max_length=6000):
    return [
        dom_content[i: i + max_length]
        for i in range(0, len(dom_content), max_length)
    ]
