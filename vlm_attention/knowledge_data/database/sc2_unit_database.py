import yaml
import json
import os
import logging
from typing import Dict, List, Any, Optional

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SC2UnitDatabase:
    """
    A class to manage and query StarCraft 2 unit data.

    This class loads unit data from a YAML index file and provides methods
    to retrieve various attributes of SC2 units.
    """

    def __init__(self, yaml_file: str):
        """
        Initialize the SC2UnitDatabase.

        Args:
            yaml_file (str): Path to the YAML index file.
        """
        self.yaml_file = yaml_file
        self.data: Dict[str, Dict[str, Dict[str, str]]] = {}
        self.unit_cache: Dict[str, Dict[str, Any]] = {}
        self.load_yaml()

    def load_yaml(self):
        """
        Load the YAML index file into memory.

        Raises:
            FileNotFoundError: If the YAML file is not found.
            yaml.YAMLError: If there's an error parsing the YAML file.
        """
        try:
            with open(self.yaml_file, 'r', encoding='utf-8') as file:
                self.data = yaml.safe_load(file)
        except FileNotFoundError:
            logging.error(f"YAML file not found: {self.yaml_file}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file: {e}")
            raise


    def get_units_by_race(self, race: str) -> List[str]:
        """
        Get a list of units for a specific race.

        Args:
            race (str): The race name (e.g., 'Protoss', 'Terran', 'Zerg').

        Returns:
            List[str]: A list of unit names for the specified race.
        """
        return list(self.data.get(race, {}).keys())

    def get_all_units_by_race(self) -> Dict[str, List[str]]:
        """
        Get all units grouped by race.

        Returns:
            Dict[str, List[str]]: A dictionary where keys are race names and values are lists of unit names.
        """
        return {race: list(units.keys()) for race, units in self.data.items()}

    def get_all_unit_names(self) -> List[str]:
        """
        Get a list of all unit names across all races.

        Returns:
            List[str]: A list of all unit names in the database.
        """
        return [unit for race_units in self.data.values() for unit in race_units]

    def unit_exists(self, unit_name: str) -> bool:
        """
        Check if a specific unit exists in the database.

        This method performs a case-insensitive check.

        Args:
            unit_name (str): The name of the unit to check.

        Returns:
            bool: True if the unit exists, False otherwise.
        """
        unit_name_lower = unit_name.lower()
        return any(unit_name_lower == name.lower() for race_units in self.data.values() for name in race_units)


    def get_unit_info(self, unit_name: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific unit.

        This method performs a case-insensitive search for the unit name.
        If found, it loads the unit's JSON file and caches the information.

        Args:
            unit_name (str): The name of the unit to search for.

        Returns:
            Dict[str, Any]: A dictionary containing the unit's information,
                            or an empty dict if the unit is not found.
        """
        if unit_name in self.unit_cache:
            return self.unit_cache[unit_name]

        for race, units in self.data.items():
            # Case-insensitive search
            unit_name_lower = unit_name.lower()
            for unit_key in units:
                if unit_name_lower == unit_key.lower():
                    file_path = units[unit_key]['file_path']
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as file:
                                unit_info = json.load(file)
                            self.unit_cache[unit_name] = unit_info
                            return unit_info
                        except json.JSONDecodeError as e:
                            logging.error(f"Error parsing JSON file for unit {unit_name}: {e}")
                    else:
                        logging.error(f"JSON file not found for unit {unit_name}: {file_path}")

        logging.warning(f"Unit information not found for: {unit_name}")
        return {}

    def search_units(self, query: str) -> List[Dict[str, str]]:
        """
        Search for units based on a query string.

        This method performs a case-insensitive partial match on unit names.

        Args:
            query (str): The search query.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing matching unit information.
                                  Each dict contains 'race', 'name', and 'file_name' keys.
        """
        results = []
        for race, units in self.data.items():
            for unit_name in units:
                if query.lower() in unit_name.lower():
                    results.append({
                        "race": race,
                        "name": unit_name,
                        "file_name": units[unit_name]['file_name']
                    })
        return results

    def get_all_races(self) -> List[str]:
        """
        Get a list of all available races.

        Returns:
            List[str]: A list of race names.
        """
        return list(self.data.keys())

    def get_unit_attribute(self, unit_name: str, attribute: str) -> Any:
        """
        Get a specific attribute for a unit.

        Args:
            unit_name (str): The name of the unit.
            attribute (str): The name of the attribute to retrieve.

        Returns:
            Any: The value of the specified attribute, or None if not found.
        """
        unit_info = self.get_unit_info(unit_name)
        if not unit_info:
            logging.warning(f"No information found for unit: {unit_name}")
            return None
        if attribute not in unit_info:
            logging.warning(f"Attribute '{attribute}' not found for unit: {unit_name}")
        return unit_info.get(attribute)

    # The following methods are specific attribute getters
    # They all use get_unit_attribute internally

    def get_unit_cost(self, unit_name: str) -> Optional[Dict[str, Any]]:
        """Get the cost information for a unit."""
        return self.get_unit_attribute(unit_name, "Cost")

    def get_unit_attack_info(self, unit_name: str) -> Optional[Dict[str, Any]]:
        """Get the attack information for a unit."""
        return self.get_unit_attribute(unit_name, "Attack")

    def get_unit_stats(self, unit_name: str) -> Optional[Dict[str, Any]]:
        """Get the stats for a unit."""
        return self.get_unit_attribute(unit_name, "Unit stats")

    def get_unit_strengths(self, unit_name: str) -> Optional[List[str]]:
        """Get the list of units that this unit is strong against."""
        return self.get_unit_attribute(unit_name, "Strong against")

    def get_unit_weaknesses(self, unit_name: str) -> Optional[List[str]]:
        """Get the list of units that this unit is weak against."""
        return self.get_unit_attribute(unit_name, "Weak against")

    def get_unit_ability(self, unit_name: str) -> Optional[str]:
        """Get the special ability of a unit."""
        return self.get_unit_attribute(unit_name, "Ability")

    def get_unit_upgrade(self, unit_name: str) -> Optional[Dict[str, Any]]:
        """Get the upgrade information for a unit."""
        return self.get_unit_attribute(unit_name, "Upgrade")

    def get_unit_competitive_usage(self, unit_name: str) -> Optional[Dict[str, str]]:
        """Get the competitive usage information for a unit."""
        return self.get_unit_attribute(unit_name, "Competitive Usage")


# Example usage
if __name__ == "__main__":
    try:
        # 初始化数据库
        db = SC2UnitDatabase("sc2_unit_data_index.yaml")

        # 测试单位列表
        test_units = [
            'Archon', 'Banshee', 'Ghost', 'Hellbat', 'Immortal',
            'Marauder', 'Marine', 'Medivac', 'Phoenix', 'Reaper',
            'Stalker', 'Viking Assault', 'Zealot'
        ]

        print("\n检查单位是否存在:")
        for unit in test_units:
            exists = db.unit_exists(unit)
            print(f"{unit}: {'存在' if exists else '不存在'}")

        print("\n尝试获取单位信息示例:")
        # 选择几个不同种族的单位进行测试
        sample_units = ['Archon', 'Marine', 'Phoenix']
        for unit in sample_units:
            info = db.get_unit_info(unit)
            print(f"\n{unit} 信息:")
            if info:
                print(f"- 找到单位信息")
                # 打印一些基本属性作为示例
                if "Unit stats" in info:
                    print(f"- 单位属性: {info['Unit stats']}")
            else:
                print(f"- 未找到单位信息")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")