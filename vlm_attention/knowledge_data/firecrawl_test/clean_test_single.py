import os
import json
from selectolax.parser import HTMLParser
import re


class SC2UnitParser:
    def __init__(self, html_content):
        self.html = HTMLParser(html_content)

    def _clean_text(self, text):
        """清理文本内容"""
        if not text:
            return ""
        # 移除编辑链接
        text = re.sub(r'\\\[[^\]]*\]', '', text)
        # 移除多余的换行和空格
        text = re.sub(r'\s+', ' ', text)
        # 清理特殊字符
        text = text.replace('\\n', ' ').replace('\\', ' ')
        return text.strip()

    def extract_description(self):
        """提取description部分"""
        try:
            # 定位Description部分:
            # 1. 首先找到包含 "Description[edit]" 的节
            # 2. 然后获取它后面的段落直到下一个section
            description = ""
            in_description = False

            for node in self.html.css('*'):
                if node.text():
                    if "Description" in node.text() and "[edit]" in node.text():
                        in_description = True
                        continue
                    elif in_description and any(
                            section in node.text() for section in ["Spells[edit]", "Abilities[edit]", "Competitive"]):
                        break
                    elif in_description and node.tag in ['p', 'div']:
                        text = self._clean_text(node.text())
                        if text and not text.startswith("edit") and not text.startswith("[edit]"):
                            description += text + " "

            return description.strip()

        except Exception as e:
            print(f"Error extracting description: {str(e)}")
            return "Error extracting description"

    def extract_spells(self):
        """提取技能信息"""
        spells = []
        try:
            in_spells = False
            current_spell = None

            for node in self.html.css('*'):
                if node.text():
                    if "Spells[edit]" in node.text():
                        in_spells = True
                        continue
                    elif in_spells and "Abilities[edit]" in node.text():
                        break

                    if in_spells:
                        if node.tag == 'table':
                            # 表格可能包含技能详细信息
                            spell_info = {}
                            for row in node.css('tr'):
                                cols = row.css('td')
                                if len(cols) >= 2:
                                    key = self._clean_text(cols[0].text())
                                    value = self._clean_text(cols[1].text())
                                    spell_info[key] = value
                            if spell_info:
                                spells.append(spell_info)

            return spells

        except Exception as e:
            print(f"Error extracting spells: {str(e)}")
            return []

    def parse(self):
        """解析所有内容"""
        return {
            "description": self.extract_description(),
            "spells": self.extract_spells(),
            "abilities": [],  # TODO
            "competitive_usage": {  # TODO
                "overview": "",
                "matchups": {}
            }
        }


def process_file(input_file, output_file):
    """处理单个文件"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'html' not in data:
            print(f"No HTML content found in {input_file}")
            return False

        parser = SC2UnitParser(data['html'])
        unit_info = parser.parse()

        # Debug输出
        print("\nExtracted description:")
        print(unit_info["description"])
        print("\nExtracted spells:")
        print(json.dumps(unit_info["spells"], indent=2))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unit_info, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return False


def main():
    """主函数"""
    input_dir = r"D:\pythoncode\vlm_attention_starcraft2\vlm_attention\knowledge_data\firecrawl_test\sc2_unit_info"
    output_dir = r"D:\pythoncode\vlm_attention_starcraft2\vlm_attention\knowledge_data\firecrawl_test\sc2_unit_info_processed_temp"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理单个文件进行测试
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            print(f"\nProcessing {filename}...")
            process_file(input_file, output_file)
            break  # 只处理第一个文件作为测试


if __name__ == "__main__":
    main()