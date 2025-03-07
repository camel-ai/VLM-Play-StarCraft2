# Liquipedia API 核心使用指南 - 信息查找与获取

## 1. API 基本结构

Liquipedia API 使用 MediaWiki API 的标准结构。基本 URL 格式如下：

```
https://liquipedia.net/api.php?action=[ACTION]&[PARAMETERS]
```

## 2. 使用限制

请求频率限制为每小时60次。
## 2. 核心参数

- **action**: 指定要执行的操作（例如：query, parse, opensearch）
- **format**: 指定输出格式（json, xml, php 等，默认为 jsonfm）
- **prop**: 在查询操作中指定要获取的属性
- **titles**: 指定要查询的页面标题
- **list**: 在查询操作中指定要获取的列表类型
- **meta**: 在查询操作中指定要获取的元数据类型

## 3. 常用查询操作

### 3.1 页面内容查询

获取指定页面的最新修订版本内容：

```python
import requests

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
```

### 3.2 分类页面查询

获取特定分类下的页面列表：

```python
import requests

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
```

### 3.3 搜索操作

使用opensearch进行搜索：

```python
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
```

### 3.4 获取页面链接

获取指定页面中的所有链接：

```python
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
```

## 4. 高级技巧

### 4.1 继续查询

当结果集很大时，API会返回一个'continue'参数。使用这个参数来获取下一批结果：

```python
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
```

### 4.2 并行请求

对于需要多次API调用的操作，可以使用并行请求来提高效率：

```python
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
```

## 5. 注意事项

1. **速率限制**: 遵守Liquipedia的API使用条款，限制请求频率（MediaWiki API每2秒1次请求）。
2. **错误处理**: 始终检查API响应中的错误信息，并适当处理。
3. **User-Agent**: 设置一个唯一的User-Agent头，包含您的项目信息和联系方式。
4. **缓存**: 合理使用缓存来减少不必要的API调用。
5. **大型数据集**: 使用continue参数来处理大型结果集。
6. **异常处理**: 在生产环境中，确保所有API调用都有适当的异常处理。

## 6. 进一步学习

- 使用 `action=help&recursivesubmodules=1` 查看完整的API文档。
- 探索Liquipedia的特定游戏API，如DOTA2、CS:GO等，它们可能有特定的端点和参数。
- 考虑使用现有的MediaWiki API包装库，如`mwclient`或`pywikibot`，它们可以简化某些操作。

通过这些示例和技巧，您应该能够有效地使用Liquipedia API进行信息查找和获取。记住始终遵守API使用条款，并尊重Liquipedia的资源。如果您有任何特定的查询需求或遇到问题，请随时寻求帮助。