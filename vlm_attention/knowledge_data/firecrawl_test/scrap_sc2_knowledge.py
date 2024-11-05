import json
from firecrawl import FirecrawlApp
import os
from datetime import datetime
import time
import requests
import random
import glob
from vlm_attention.knowledge_data import url

# 配置
CONFIG = {
    "api_key": "fc-96cdb2de30954131b1a9e6cc8bb34d65",
    "proxy": {
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890"
    },
    "scrape_params": {
        "formats": ["markdown", "html"],
        "metadata": True  # 确保获取元数据
    },
    "output_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "sc2_unit_info"),
    "max_retries": 5,
    "retry_delay": 15,  # 秒
    "error_log_file": "scraping_errors.log"
}


def setup_proxy(proxy_config):
    """设置代理配置"""
    for protocol, url in proxy_config.items():
        os.environ[f'{protocol.upper()}_PROXY'] = url


def save_result(data, filename):
    """保存结果到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {filename}")


def log_error(message):
    """记录错误信息"""
    with open(CONFIG["error_log_file"], 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now()}: {message}\n")


def validate_unit_data(unit_data):
    """验证爬取的数据是否完整"""
    validation_issues = []
    for unit_name, data in unit_data.items():
        if not data.get('html'):
            validation_issues.append(f"Missing HTML content for unit {unit_name}")
        if not data.get('markdown'):
            validation_issues.append(f"Missing Markdown content for unit {unit_name}")
        if not data.get('metadata', {}).get('title'):
            validation_issues.append(f"Missing title metadata for unit {unit_name}")
        if not data.get('metadata', {}).get('description'):
            validation_issues.append(f"Missing description metadata for unit {unit_name}")

    return validation_issues


def format_scrape_result(raw_result, url):
    """格式化爬取结果为预期格式"""
    # 优先使用 html 字段，如果不存在则使用 content 字段
    html_content = raw_result.get("html", raw_result.get("content", ""))

    formatted_result = {
        "markdown": raw_result.get("markdown", ""),
        "html": html_content,  # 使用获取到的 HTML 内容
        "metadata": {
            "title": raw_result.get("metadata", {}).get("title", ""),
            "description": raw_result.get("metadata", {}).get("description", ""),
            "language": raw_result.get("metadata", {}).get("language", "en"),
            "ogTitle": raw_result.get("metadata", {}).get("og:title", ""),
            "ogDescription": raw_result.get("metadata", {}).get("og:description", ""),
            "ogUrl": raw_result.get("metadata", {}).get("og:url", ""),
            "ogImage": raw_result.get("metadata", {}).get("og:image", ""),
            "ogLocaleAlternate": raw_result.get("metadata", {}).get("og:locale:alternate", []),
            "ogSiteName": raw_result.get("metadata", {}).get("og:site_name", ""),
            "sourceURL": url,
            "statusCode": raw_result.get("status_code", 200)
        }
    }
    return formatted_result


def scrape_unit_with_retry(app, race, unit_name, unit_url):
    """使用重试机制爬取单个单位信息"""
    for attempt in range(CONFIG["max_retries"]):
        try:
            print(f"Scraping unit: {unit_name} (Attempt {attempt + 1})")
            raw_result = app.scrape_url(unit_url, params=CONFIG["scrape_params"])

            # 格式化结果
            formatted_result = format_scrape_result(raw_result, unit_url)

            # 保存单个单位的信息
            unit_filename = os.path.join(CONFIG["output_dir"], f"{race}_{unit_name}.json")
            save_result(formatted_result, unit_filename)

            return {unit_name: formatted_result}

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limit exceeded. Waiting for {CONFIG['retry_delay']} seconds before retrying...")
                time.sleep(CONFIG['retry_delay'])
            elif e.response.status_code == 404:
                error_msg = f"URL not found for {unit_name}: {unit_url}"
                print(error_msg)
                log_error(error_msg)
                return {unit_name: {
                    "markdown": "",
                    "html": "",
                    "metadata": {
                        "statusCode": 404,
                        "sourceURL": unit_url,
                        "error": "404 Not Found"
                    }
                }}
            else:
                error_msg = f"HTTP error occurred for {unit_name}: {e}"
                print(error_msg)
                log_error(error_msg)
                if attempt == CONFIG["max_retries"] - 1:
                    return {unit_name: {
                        "markdown": "",
                        "html": "",
                        "metadata": {
                            "statusCode": e.response.status_code,
                            "sourceURL": unit_url,
                            "error": str(e)
                        }
                    }}
        except Exception as e:
            error_msg = f"An error occurred while scraping {unit_name}: {e}"
            print(error_msg)
            log_error(error_msg)
            if attempt == CONFIG["max_retries"] - 1:
                return {unit_name: {
                    "markdown": "",
                    "html": "",
                    "metadata": {
                        "statusCode": 500,
                        "sourceURL": unit_url,
                        "error": str(e)
                    }
                }}

        time.sleep(CONFIG['retry_delay'])

    error_msg = f"Failed to scrape {unit_name} after {CONFIG['max_retries']} attempts"
    print(error_msg)
    log_error(error_msg)
    return {unit_name: {
        "markdown": "",
        "html": "",
        "metadata": {
            "statusCode": 500,
            "sourceURL": unit_url,
            "error": "Max retries reached"
        }
    }}


def test_single_url():
    """测试函数，用于调试API返回的具体内容"""
    app = FirecrawlApp(api_key=CONFIG["api_key"])
    setup_proxy(CONFIG["proxy"])

    test_url = "https://liquipedia.net/starcraft2/Archon_(Legacy_of_the_Void)"

    print("Testing with current config:")
    print(json.dumps(CONFIG["scrape_params"], indent=2))

    result = app.scrape_url(test_url, params=CONFIG["scrape_params"])
    print("\nResponse contains:")
    for key in result.keys():
        print(f"- {key}")

    print("\nContent details:")
    if 'content' in result:
        print(f"Content field length: {len(result['content'])}")
    if 'html' in result:
        print(f"HTML field length: {len(result['html'])}")

    # 测试格式化结果
    formatted = format_scrape_result(result, test_url)
    print("\nFormatted result HTML length:", len(formatted['html']))

    return result, formatted


def main():
    """主函数：爬取所有单位信息"""
    app = FirecrawlApp(api_key=CONFIG["api_key"])
    setup_proxy(CONFIG["proxy"])

    # 创建输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    all_units_data = {}
    total_units = sum(len(units) for units in url.knowledge_url["Unit"].values())
    processed_units = 0
    start_time = time.time()

    for race, units in url.knowledge_url["Unit"].items():
        print(f"\nProcessing {race} units ({len(units)} units):")
        all_units_data[race] = {}

        for unit_name, unit_url in units.items():
            processed_units += 1
            print(f"[{processed_units}/{total_units}] Scraping {race} unit: {unit_name}")

            unit_data = scrape_unit_with_retry(app, race, unit_name, unit_url)
            all_units_data[race].update(unit_data)

            # 在每次请求之间增加随机延迟（1.5-3.5秒）
            delay = 1.5 + random.random() * 2
            time.sleep(delay)

    # 计算总耗时
    elapsed_time = time.time() - start_time

    # 保存所有单位的汇总信息
    consolidated_filename = os.path.join(CONFIG["output_dir"],
                                         f"all_units_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_result(all_units_data, consolidated_filename)

    print(f"\nScraping completed. Total units processed: {processed_units}")
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")
    print(f"Average time per unit: {elapsed_time / processed_units:.2f} seconds")
    print(f"Results saved to: {consolidated_filename}")

    # 验证数据
    print("\nValidating scraped data...")
    for race, units in all_units_data.items():
        print(f"\nValidating {race} units:")
        issues = validate_unit_data(units)
        if issues:
            print("\n".join(issues))
        else:
            print(f"All {race} units data validated successfully!")

    print(f"\nCheck {CONFIG['error_log_file']} for any errors during scraping.")


if __name__ == "__main__":
    # test_single_url()
    main()