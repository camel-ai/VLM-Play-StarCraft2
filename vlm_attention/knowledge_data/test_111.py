from firecrawl.firecrawl import FirecrawlApp
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
app = FirecrawlApp(api_key="fc-96cdb2de30954131b1a9e6cc8bb34d65")

# Scrape a website:
scrape_status = app.scrape_url(
  'https://docs.firecrawl.dev',
  params={'formats': ['markdown', 'html']}
)
print(scrape_status)

# Crawl a website:
crawl_status = app.crawl_url(
  'https://docs.firecrawl.dev',
  params={
    'limit': 100,
    'scrapeOptions': {'formats': ['markdown', 'html']}
  },
  poll_interval=30
)
print(crawl_status)