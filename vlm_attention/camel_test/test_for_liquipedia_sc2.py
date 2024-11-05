import requests
import os
import json

# 设置代理（如果需要的话）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# Liquipedia StarCraft II API 基础 URL
BASE_URL = "https://liquipedia.net/starcraft2/api.php"


def get_sc2_page_content(title):
    """获取指定页面的内容"""
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": title,
        "rvprop": "content",
        "rvslots": "main"
    }
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # 如果请求失败，这将引发一个异常
        data = response.json()

        # 打印完整的 JSON 响应，用于调试
        print(f"API Response for '{title}':")
        print(json.dumps(data, indent=2))

        page = next(iter(data["query"]["pages"].values()))
        if "revisions" not in page:
            print(f"Warning: No revisions found for '{title}'. Page might not exist.")
            return None
        content = page["revisions"][0]["slots"]["main"]["*"]
        return content
    except requests.RequestException as e:
        print(f"Error fetching content for '{title}': {e}")
        return None


def get_sc2_category_members(category, limit=10):
    """获取指定分类的成员"""
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": str(limit)
    }
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        return [page["title"] for page in data["query"]["categorymembers"]]
    except requests.RequestException as e:
        print(f"Error fetching category members for '{category}': {e}")
        return []


def search_sc2_liquipedia(query, limit=10):
    """搜索星际争霸2相关内容"""
    params = {
        "action": "opensearch",
        "format": "json",
        "search": query,
        "limit": str(limit)
    }
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        return list(zip(data[1], data[3]))
    except requests.RequestException as e:
        print(f"Error searching for '{query}': {e}")
        return []


def get_sc2_player_info(player_name):
    """获取指定选手的信息"""
    content = get_sc2_page_content(player_name)
    if content:
        return content[:500]  # 为了演示，我们只返回前500个字符
    return "Player information not found."


def get_sc2_tournament_info(tournament_name):
    """获取指定锦标赛的信息"""
    content = get_sc2_page_content(tournament_name)
    if content:
        return content[:500]  # 为了演示，我们只返回前500个字符
    return "Tournament information not found."


# 演示使用
if __name__ == "__main__":
    print("StarCraft II Liquipedia API Demo\n")

    # 1. 获取StarCraft II页面内容
    print("1. StarCraft II 页面内容:")
    sc2_content = get_sc2_page_content("StarCraft II")
    print(sc2_content[:300] if sc2_content else "Content not found.")
    print("\n" + "-" * 50 + "\n")

    # 2. 获取StarCraft II职业选手列表
    print("2. StarCraft II 职业选手列表:")
    sc2_players = get_sc2_category_members("StarCraft II players", limit=5)
    for player in sc2_players:
        print(player)
    print("\n" + "-" * 50 + "\n")

    # 3. 搜索StarCraft II相关内容
    print("3. 搜索 'GSL' 相关内容:")
    search_results = search_sc2_liquipedia("GSL", limit=5)
    for title, url in search_results:
        print(f"{title}: {url}")
    print("\n" + "-" * 50 + "\n")

    # 4. 获取特定选手信息
    print("4. Serral 选手信息:")
    serral_info = get_sc2_player_info("Serral")
    print(serral_info)
    print("\n" + "-" * 50 + "\n")

    # 5. 获取特定锦标赛信息
    print("5. IEM Katowice 2023 锦标赛信息:")
    iem_info = get_sc2_tournament_info("IEM Katowice 2023")
    print(iem_info)