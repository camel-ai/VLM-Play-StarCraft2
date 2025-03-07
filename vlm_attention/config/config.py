"""
config the openai api key and model name
"""

config_dir = {
    "openai": {
        "api_key": "your key",
        "organization_name": "CAMEL AI",
        "organization_id": "your organization id",
        "llm_model_name": "gpt-4o-mini",
        "vlm_model_name": "gpt-4o-mini"
    },
    "qwen": {
        "api_key": "your key",  # 这里填写您的通义千问API密钥
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "llm_model_name": "qwen-plus",  # 添加这个
        "vlm_model_name": "qvq-72b-preview"  # 添加这个
    },
    "proxy": {
        "port": 7890,
        "url": "http://127.0.0.1:7890"
    },
    "current_model": "openai"

    # 其他配置类别可以在这里添加
}







