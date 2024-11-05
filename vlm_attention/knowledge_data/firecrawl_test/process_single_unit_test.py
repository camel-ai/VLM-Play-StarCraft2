import json
from bs4 import BeautifulSoup
import os

class SC2UnitInfoExtractor:
    def __init__(self, html_content):
        self.soup = BeautifulSoup(html_content, 'html.parser')

    def extract_unit_name(self):
        name_element = self.soup.find('h1', id='firstHeading')
        if name_element:
            span_element = name_element.find('span', class_='mw-page-title-main')
            if span_element:
                return span_element.text.strip()
            else:
                print("Debug: mw-page-title-main span not found")
                return name_element.text.strip()
        print("Debug: firstHeading not found")
        return 'N/A'

    def extract_unit_information(self):
        unit_info = {}
        info_section = self.soup.find('div', class_='infobox-header', string='Unit Information')
        if info_section:
            print("Debug: Unit Information section found")
            current_div = info_section.find_next_sibling('div')
            while current_div and 'infobox-header' not in current_div.get('class', []):
                if 'infobox-cell-2' in current_div.get('class', []):
                    key = current_div.text.strip().rstrip(':')
                    value_div = current_div.find_next_sibling('div')
                    if value_div:
                        if key in ['Requirements', 'Strong against', 'Weak against']:
                            value = [li.text.strip() for li in value_div.find_all('li')]
                        elif key == 'Cost':
                            value = ' '.join(value_div.stripped_strings)
                        else:
                            value = value_div.text.strip()
                        unit_info[key] = value
                current_div = current_div.find_next_sibling('div')
        else:
            print("Debug: Unit Information section not found")
        return unit_info

    def extract_attack_info(self):
        attack_info = {}
        attack_section = self.soup.find('div', class_='infobox-header', string='Attack 1: Psi Blast')
        if attack_section:
            print("Debug: Attack section found")
            current_div = attack_section.find_next_sibling('div')
            while current_div and 'infobox-header' not in current_div.get('class', []):
                if 'infobox-cell-2' in current_div.get('class', []):
                    key = current_div.text.strip().rstrip(':')
                    value_div = current_div.find_next_sibling('div')
                    if value_div:
                        attack_info[key] = value_div.text.strip()
                current_div = current_div.find_next_sibling('div')
        else:
            print("Debug: Attack section not found")
        return attack_info

    def extract_unit_stats(self):
        unit_stats = {}
        stats_section = self.soup.find('div', class_='infobox-header', string='Unit stats')
        if stats_section:
            print("Debug: Unit stats section found")
            current_div = stats_section.find_next_sibling('div')
            while current_div:
                if 'infobox-cell-2' in current_div.get('class', []):
                    key = current_div.text.strip().rstrip(':')
                    value_div = current_div.find_next_sibling('div')
                    if value_div:
                        if key in ['Strong against', 'Weak against']:
                            value = [li.text.strip() for li in value_div.find_all('li')]
                        else:
                            value = ' '.join(value_div.stripped_strings)
                        unit_stats[key] = value
                current_div = current_div.find_next_sibling('div')
        else:
            print("Debug: Unit stats section not found")
        return unit_stats

    def extract_all_info(self):
        return {
            'name': self.extract_unit_name(),
            'unit_information': self.extract_unit_information(),
            'attack_info': self.extract_attack_info(),
            'unit_stats': self.extract_unit_stats()
        }

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    html_content = data['html']

    if not html_content:
        print("Error: HTML content is empty")
        return

    print("Debug: HTML content length:", len(html_content))
    print("Debug: First 500 characters of HTML:")
    print(html_content[:500])

    extractor = SC2UnitInfoExtractor(html_content)
    unit_info = extractor.extract_all_info()

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unit_info, f, ensure_ascii=False, indent=2)

    print("Extracted information:")
    print(json.dumps(unit_info, indent=2))

def main():
    input_file = r"C:\python_code\vlm_attention_starcraft2-main\vlm_attention\knowledge_data\firecrawl_test\sc2_unit_info\Protoss_high_templar.json"
    output_file = r"C:\python_code\vlm_attention_starcraft2-main\vlm_attention\knowledge_data\firecrawl_test\sc2_unit_info_processed\Protoss_high_templar_processed.json"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    process_file(input_file, output_file)
    print(f"Processed: {os.path.basename(input_file)}")

if __name__ == "__main__":
    main()