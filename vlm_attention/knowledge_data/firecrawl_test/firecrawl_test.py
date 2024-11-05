import json
from firecrawl import FirecrawlApp
import os
from datetime import datetime
import argparse

# Default configuration
DEFAULT_CONFIG = {
    "api_key": "fc-96cdb2de30954131b1a9e6cc8bb34d65",
    "proxy": {
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890"
    },
    "target_url": "https://liquipedia.net/starcraft2/Micro_(StarCraft)",
    "scrape_params": {
        "formats": ["markdown", "html"]
    },
    "crawl_params": {
        "limit": 100,
        "scrapeOptions": {
            "formats": ["markdown", "html"]
        }
    }
}


def save_result(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {filename}")


def setup_proxy(proxy_config):
    for protocol, url in proxy_config.items():
        os.environ[f'{protocol.upper()}_PROXY'] = url


def scrape_website(app, url, params):
    print(f"Scraping website: {url}")
    scrape_status = app.scrape_url(url, params=params)
    save_result(scrape_status, f'scrape_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')


def crawl_website(app, url, params):
    print(f"Crawling website: {url}")
    crawl_status = app.crawl_url(url, params=params)
    save_result(crawl_status, f'crawl_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')


def main(config):
    app = FirecrawlApp(api_key=config["api_key"])

    setup_proxy(config["proxy"])

    scrape_website(app, config["target_url"], config["scrape_params"]) # 抓取单个页面
    crawl_website(app, config["target_url"], config["crawl_params"]) # 抓取整个网站


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Firecrawl script with flexible configuration")
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG

    main(config)