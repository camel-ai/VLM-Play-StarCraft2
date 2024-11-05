import requests
import time
import random
from bs4 import BeautifulSoup
from datetime import datetime
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fake_useragent import UserAgent
import urllib3
import os

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SC2TournamentScraper:
    def __init__(self, use_proxy=False):
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
        self.base_url = "https://liquipedia.net"
        self.api_url = "https://liquipedia.net/starcraft2/api.php"
        self.session = requests.Session()
        self.ua = UserAgent()

        # Disable SSL verification
        self.session.verify = False

        # Implement a more conservative retry strategy
        retries = Retry(
            total=10,
            backoff_factor=5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

        if use_proxy:
            self.session.proxies = {
                'http': 'http://127.0.0.1:7890',
                'https': 'http://127.0.0.1:7890',
            }

    def update_headers(self):
        self.session.headers.update({
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        })

    def make_request(self, url, params=None, retries=5):
        self.update_headers()
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()

                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    wait_time = retry_after * (2 ** attempt)  # Exponential backoff
                    logging.warning(f"Rate limited. Attempt {attempt + 1}/{retries}. Waiting for {wait_time} seconds.")
                    time.sleep(wait_time)
                    continue

                time.sleep(random.uniform(10, 20))  # Increased random delay
                return response
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
                if attempt < retries - 1:
                    wait_time = (2 ** attempt) * 10
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return None
        return None

    def get_page_content(self, title):
        params = {
            "action": "parse",
            "page": title,
            "format": "json"
        }
        response = self.make_request(self.api_url, params)
        if response:
            data = response.json()
            return data['parse']['text']['*']
        return None

    def parse_date(self, date_string):
        if not date_string or date_string.isspace():
            return None

        date_string = date_string.split('to')[-1].strip()  # Use end date if it's a range
        date_formats = ['%Y-%m-%d', '%B %d, %Y', '%d %B %Y', '%Y-%m-%d %H:%M', '%B %d, %Y %H:%M']
        for fmt in date_formats:
            try:
                return datetime.strptime(date_string, fmt).date()
            except ValueError:
                continue

        logging.warning(f"Unable to parse date '{date_string}'")
        return None

    def get_tournament_infobox(self, soup):
        tournament = {}
        info_boxes = soup.find_all('div', class_='infobox-cell-2')
        for i in range(0, len(info_boxes), 2):
            attribute = info_boxes[i].get_text().replace(':', '').strip()
            value = info_boxes[i + 1].get_text().strip()

            if attribute == "League":
                tournament['name'] = value
            elif attribute == "Date":
                tournament['date'] = self.parse_date(value)
            elif attribute == "Prize Pool":
                tournament['prize_pool'] = value
            elif attribute == "Teams" or attribute == "Players":
                tournament['participants'] = value
            elif attribute == "Location":
                tournament['location'] = value
            elif attribute == "Organizer":
                tournament['organizer'] = value

        return tournament

    def get_tournament_results(self, soup):
        results = []
        results_table = soup.find('table', class_='wikitable')
        if results_table:
            rows = results_table.find_all('tr')
            for row in rows[1:]:  # Skip header row
                cells = row.find_all('td')
                if len(cells) >= 3:
                    result = {
                        'place': cells[0].get_text().strip(),
                        'team_or_player': cells[1].get_text().strip(),
                        'prize': cells[2].get_text().strip()
                    }
                    results.append(result)
        return results

    def get_tournament_data(self, tournament_name):
        content = self.get_page_content(tournament_name)
        if not content:
            return None

        soup = BeautifulSoup(content, 'html.parser')

        tournament_data = self.get_tournament_infobox(soup)
        tournament_data['results'] = self.get_tournament_results(soup)

        return tournament_data

    def get_recent_tournaments(self, limit=10):
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": "Category:Tournaments",
            "cmsort": "timestamp",
            "cmdir": "desc",
            "cmlimit": str(limit)
        }
        response = self.make_request(self.api_url, params)
        if not response:
            return []

        data = response.json()
        tournaments = []
        for page in data["query"]["categorymembers"]:
            tournament_data = self.get_tournament_data(page["title"])
            if tournament_data:
                tournaments.append(tournament_data)
                if len(tournaments) == 5:  # 获取5个有效的赛事
                    break
        return tournaments


def main():
    scraper = SC2TournamentScraper(use_proxy=True)
    print("StarCraft II Tournament Data Scraper\n")

    try:
        recent_tournaments = scraper.get_recent_tournaments(limit=3)  # Further reduced limit to 3
        print("Recent Tournaments:")
        for tournament in recent_tournaments:
            print(f"- {tournament['name']} (Date: {tournament['date']})")
            print(f"  Prize Pool: {tournament['prize_pool']}")
            print(f"  Participants: {tournament.get('participants', 'N/A')}")
            print(f"  Top 3 Results:")
            for result in tournament['results'][:3]:
                print(f"    {result['place']}. {result['team_or_player']} - {result['prize']}")
            print()

        if recent_tournaments:
            print("\nDetailed information for the most recent tournament:")
            most_recent = recent_tournaments[0]
            print(f"Name: {most_recent['name']}")
            print(f"Date: {most_recent['date']}")
            print(f"Prize Pool: {most_recent['prize_pool']}")
            print(f"Location: {most_recent.get('location', 'N/A')}")
            print(f"Organizer: {most_recent.get('organizer', 'N/A')}")
            print(f"Participants: {most_recent.get('participants', 'N/A')}")
            print("\nFull Results:")
            for result in most_recent['results']:
                print(f"  {result['place']}. {result['team_or_player']} - {result['prize']}")
        else:
            print("No recent tournaments found.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print("An error occurred while fetching tournament knowledge_data. Please check the logs for more information.")

if __name__ == "__main__":
    main()