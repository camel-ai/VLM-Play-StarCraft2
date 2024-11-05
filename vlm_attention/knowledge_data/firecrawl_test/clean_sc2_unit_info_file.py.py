import os
import json
from bs4 import BeautifulSoup
import re


class SC2UnitParser:
    def __init__(self, html_content):
        self.soup = BeautifulSoup(html_content, 'html.parser')

    def _get_section_text(self, start_text, end_text=None):
        """Helper to get text between two section headers."""
        start_pattern = re.compile(f"{start_text}\\\\")
        start_elem = self.soup.find(string=start_pattern)
        if not start_elem:
            return ""

        content = []
        current = start_elem.find_parent().find_next_sibling()

        while current:
            # 如果遇到下一个section，停止
            if current.name in ['h2', 'h3'] or (end_text and end_text in str(current)):
                break
            if current.name == 'p':
                text = current.get_text(strip=True)
                if text and not text.startswith('\\['):  # 排除编辑链接
                    content.append(text)
            current = current.next_sibling

        return " ".join(content)

    def extract_description(self):
        """Extract description section content."""
        description = self._get_section_text("Description", "Abilities")
        return description if description else "Not found"

    def _parse_table_row(self, row):
        """Helper to parse table rows with two columns."""
        cells = row.find_all(['td'])
        if len(cells) == 2:
            return cells[1].get_text(strip=True)
        return None

    def extract_abilities(self):
        """Extract abilities information."""
        abilities = []
        ability_section = self.soup.find(string=re.compile("Abilities\\\\"))
        if ability_section:
            current = ability_section.find_parent().find_next_sibling()

            while current and not (current.name == 'h2' or "Upgrades\\" in str(current)):
                if current.name == 'p':
                    ability_name = current.get_text(strip=True)
                elif current.name == 'table':
                    ability = {'name': ability_name}
                    for row in current.find_all('tr'):
                        text = self._parse_table_row(row)
                        if text:
                            if 'Minerals' in text:
                                ability['mineral_cost'] = int(re.search(r'\d+', text).group())
                            elif 'Gas' in text:
                                ability['gas_cost'] = int(re.search(r'\d+', text).group())
                            elif 'Duration' in text:
                                ability['duration'] = float(re.search(r'\d+', text).group())
                            elif 'Hotkey' in text:
                                ability['hotkey'] = text.split(':')[-1].strip()
                    abilities.append(ability)
                current = current.next_sibling
        return abilities

    def extract_upgrades(self):
        """Extract upgrades information."""
        upgrades = []
        upgrade_section = self.soup.find(string=re.compile("Upgrades\\\\"))
        if upgrade_section:
            current = upgrade_section.find_parent().find_next_sibling()

            while current and not (current.name == 'h2' or "Competitive Usage\\" in str(current)):
                if current.name == 'p':
                    upgrade_name = current.get_text(strip=True)
                elif current.name == 'table':
                    upgrade = {'name': upgrade_name}
                    for row in current.find_all('tr'):
                        text = self._parse_table_row(row)
                        if text:
                            if 'Minerals' in text:
                                upgrade['mineral_cost'] = int(re.search(r'\d+', text).group())
                            elif 'Gas' in text:
                                upgrade['gas_cost'] = int(re.search(r'\d+', text).group())
                            elif 'Game Speed' in text:
                                upgrade['duration'] = float(re.search(r'\d+', text).group())
                            elif 'Hotkey' in text:
                                upgrade['hotkey'] = text.split(':')[-1].strip()
                            elif 'Researched from' in text:
                                upgrade['research_building'] = text.split(':')[-1].strip()
                    upgrades.append(upgrade)
                current = current.next_sibling
        return upgrades

    def extract_competitive_usage(self):
        """Extract competitive usage information."""
        usage = {
            "overview": "",
            "matchups": {}
        }

        # Extract overview
        usage["overview"] = self._get_section_text("Competitive Usage", "Vs.")

        # Extract matchup information
        for race in ['Protoss', 'Terran', 'Zerg']:
            matchup_text = self._get_section_text(f"Vs..*{race}")
            if matchup_text:
                usage["matchups"][f"vs_{race}"] = matchup_text

        return usage

    def parse(self):
        return {
            "description": self.extract_description(),
            "abilities": self.extract_abilities(),
            "upgrades": self.extract_upgrades(),
            "competitive_usage": self.extract_competitive_usage()
        }


def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    html_content = data['html']
    parser = SC2UnitParser(html_content)
    unit_info = parser.parse()

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unit_info, f, indent=2, ensure_ascii=False)


def main():
    input_dir = r"D:\pythoncode\vlm_attention_starcraft2\vlm_attention\knowledge_data\firecrawl_test\sc2_unit_info"
    output_dir = r"D:\pythoncode\vlm_attention_starcraft2\vlm_attention\knowledge_data\firecrawl_test\sc2_unit_info_processed_temp"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            print(f"Processing {filename}...")
            try:
                process_file(input_file, output_file)
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    main()