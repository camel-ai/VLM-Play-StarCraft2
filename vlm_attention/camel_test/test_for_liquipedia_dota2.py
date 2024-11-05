import requests
import os


# Set up proxy
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

def get_page_content(title):
    PARAMS = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": title,
        "rvprop": "content",
        "rvslots": "main"
    }

    response = requests.get("https://liquipedia.net/dota2/api.php", params=PARAMS)
    data = response.json()

    page = next(iter(data["query"]["pages"].values()))
    content = page["revisions"][0]["slots"]["main"]["*"]
    return content


# 使用示例
dota2_content = get_page_content("Dota 2")
print(dota2_content[:500])  # 打印前500个字符

import requests

print("_____________________________")

def get_category_members(category, limit=10):
    PARAMS = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": str(limit)
    }

    response = requests.get("https://liquipedia.net/dota2/api.php", params=PARAMS)
    data = response.json()

    return [page["title"] for page in data["query"]["categorymembers"]]


# 使用示例
dota2_teams = get_category_members("Dota 2 Teams")
print(dota2_teams)

print("_____________________________")

import requests


def search_liquipedia(query, limit=10):
    PARAMS = {
        "action": "opensearch",
        "format": "json",
        "search": query,
        "limit": str(limit)
    }

    response = requests.get("https://liquipedia.net/dota2/api.php", params=PARAMS)
    data = response.json()

    return list(zip(data[1], data[3]))  # 返回标题和URL的列表


# 使用示例
search_results = search_liquipedia("The International")
for title, url in search_results:
    print(f"{title}: {url}")

print("_____________________________")
import requests


def get_page_links(title):
    PARAMS = {
        "action": "query",
        "format": "json",
        "prop": "links",
        "titles": title,
        "pllimit": "max"
    }

    response = requests.get("https://liquipedia.net/dota2/api.php", params=PARAMS)
    data = response.json()

    page = next(iter(data["query"]["pages"].values()))
    return [link["title"] for link in page.get("links", [])]


# 使用示例
dota2_links = get_page_links("Dota 2")
print(dota2_links[:20])  # 打印前20个链接

print("_____________________________")
import requests


def get_all_category_members(category):
    PARAMS = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": "max"
    }

    members = []
    while True:
        response = requests.get("https://liquipedia.net/dota2/api.php", params=PARAMS)
        data = response.json()

        members.extend([page["title"] for page in data["query"]["categorymembers"]])

        if "continue" not in data:
            break

        PARAMS.update(data["continue"])

    return members


# 使用示例
all_dota2_teams = get_all_category_members("Dota 2 Teams")
print(f"Total teams: {len(all_dota2_teams)}")
print(all_dota2_teams[:20])  # 打印前20个团队

print("_____________________________")
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_page_info(title):
    PARAMS = {
        "action": "query",
        "format": "json",
        "prop": "info|revisions",
        "titles": title,
        "rvprop": "timestamp|user",
        "rvlimit": "1"
    }

    response = requests.get("https://liquipedia.net/dota2/api.php", params=PARAMS)
    data = response.json()

    page = next(iter(data["query"]["pages"].values()))
    return {
        "title": page["title"],
        "last_edit": page["revisions"][0]["timestamp"],
        "last_editor": page["revisions"][0]["user"],
        "length": page["length"]
    }


def get_multiple_page_info(titles):
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_title = {executor.submit(get_page_info, title): title for title in titles}
        results = {}
        for future in as_completed(future_to_title):
            title = future_to_title[future]
            try:
                results[title] = future.result()
            except Exception as exc:
                print(f'{title} generated an exception: {exc}')
    return results


# 使用示例
titles_to_check = ["Dota 2", "The International", "Alliance", "OG"]
page_infos = get_multiple_page_info(titles_to_check)
for title, info in page_infos.items():
    print(f"{title}: Last edited by {info['last_editor']} on {info['last_edit']}, length: {info['length']}")