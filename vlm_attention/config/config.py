config = {
    "openai": {
        "api_key": "sk-hiP_PDdVZTqFkXRZ6YE2FDRHojCl98uO2bxZboHul2T3BlbkFJ5IKXSHi1721jZ8buILxO0kP0KFpLnqyk99wlhqG_gA",
        "organization_name": "CAMEL AI",
        "organization_id": "org-YBLeiIenLZxY3ncCUDNLT67O",
        "llm_model_name": "gpt-4o-mini-2024-07-18",
        "vlm_model_name":"gpt-4o-mini-2024-07-18"
    },
    "proxy": {
        "port": 7890,
        "url": "http://127.0.0.1:7890"
    },

    # 其他配置类别可以在这里添加
}


# 如果需要，可以添加一些辅助函数来获取配置值
def get_config(category, key):
    return config.get(category, {}).get(key)


# 使用示例
if __name__ == "__main__":
    print(get_config("openai", "api_key"))
    print(get_config("proxy", "url"))
