# Liquipedia API Core Usage Guide - Information Search and Retrieval

## 1. API Basic Structure

The Liquipedia API uses the standard structure of the MediaWiki API. The basic URL format is as follows:

```
https://liquipedia.net/api.php?action=[ACTION]&[PARAMETERS]
```

## 2. Usage Limitations

Request frequency is limited to 60 times per hour.

## 2. Core Parameters

- **action**: Specifies the operation to perform (e.g., query, parse, opensearch)
- **format**: Specifies the output format (json, xml, php, etc., default is jsonfm)
- **prop**: Specifies the properties to retrieve in a query operation
- **titles**: Specifies the page titles to query
- **list**: Specifies the type of list to retrieve in a query operation
- **meta**: Specifies the type of metadata to retrieve in a query operation

## 3. Common Query Operations

### 3.1 Page Content Query

Get the content of the latest revision of a specified page:

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

# Usage example
dota2_content = get_page_content("Dota 2")
print(dota2_content[:500])  # Print the first 500 characters
```

### 3.2 Category Page Query

Get a list of pages in a specific category:

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

# Usage example
dota2_teams = get_category_members("Dota 2 Teams")
print(dota2_teams)
```

### 3.3 Search Operations

Use opensearch to perform searches:

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
  
    return list(zip(data[1], data[3]))  # Return a list of titles and URLs

# Usage example
search_results = search_liquipedia("The International")
for title, url in search_results:
    print(f"{title}: {url}")
```

### 3.4 Get Page Links

Get all links from a specified page:

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

# Usage example
dota2_links = get_page_links("Dota 2")
print(dota2_links[:20])  # Print the first 20 links
```

## 4. Advanced Techniques

### 4.1 Continue Query

When the result set is large, the API will return a 'continue' parameter. Use this parameter to get the next batch of results:

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

# Usage example
all_dota2_teams = get_all_category_members("Dota 2 Teams")
print(f"Total teams: {len(all_dota2_teams)}")
print(all_dota2_teams[:20])  # Print the first 20 teams
```

### 4.2 Parallel Requests

For operations that require multiple API calls, parallel requests can be used to improve efficiency:

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

# Usage example
titles_to_check = ["Dota 2", "The International", "Alliance", "OG"]
page_infos = get_multiple_page_info(titles_to_check)
for title, info in page_infos.items():
    print(f"{title}: Last edited by {info['last_editor']} on {info['last_edit']}, length: {info['length']}")
```

## 5. Important Considerations

1. **Rate Limits**: Adhere to Liquipedia's API terms of use, limiting request frequency (MediaWiki API 1 request per 2 seconds).
2. **Error Handling**: Always check for error messages in API responses and handle them appropriately.
3. **User-Agent**: Set a unique User-Agent header that includes your project information and contact details.
4. **Caching**: Use caching sensibly to reduce unnecessary API calls.
5. **Large Datasets**: Use the continue parameter to handle large result sets.
6. **Exception Handling**: Ensure all API calls have appropriate exception handling in production environments.

## 6. Further Learning

- Use `action=help&recursivesubmodules=1` to view the complete API documentation.
- Explore Liquipedia's game-specific APIs, such as DOTA2, CS:GO, etc., which may have specific endpoints and parameters.
- Consider using existing MediaWiki API wrapper libraries, such as `mwclient` or `pywikibot`, which can simplify certain operations.

With these examples and techniques, you should be able to effectively use the Liquipedia API for information search and retrieval. Remember to always comply with the API terms of use and respect Liquipedia's resources. If you have any specific query needs or encounter problems, feel free to seek help.
