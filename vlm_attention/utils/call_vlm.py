import base64
import os
from openai import OpenAI
from vlm_attention.config.config import get_config
import logging
from logging.handlers import RotatingFileHandler


def setup_logger(name, log_file, level=logging.INFO):
    """设置一个指定名称的日志记录器"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


# 设置主日志记录器
main_logger = setup_logger('main_logger', 'chatbot.log', level=logging.DEBUG)


class BaseChatbot:
    def __init__(self, model_name):
        # 设置代理
        proxy_url = get_config("proxy", "url")
        os.environ["http_proxy"] = proxy_url
        os.environ["https_proxy"] = proxy_url
        main_logger.debug(f"Proxy set to: {proxy_url}")

        # 创建 OpenAI 客户端
        self.client = OpenAI(
            api_key=get_config("openai", "api_key"),
            organization=get_config("openai", "organization_id")
        )
        main_logger.debug("OpenAI client created")

        self.model = model_name
        main_logger.debug(f"Model set to: {self.model}")
        self.conversation_history = []

    def query(self, system_prompt, user_input):
        raise NotImplementedError("Subclasses must implement this method")

    def clear_history(self):
        self.conversation_history = []


class TextChatbot(BaseChatbot):
    def query(self, system_prompt, user_input, maintain_history=False):
        try:
            messages = [{"role": "system", "content": system_prompt}]

            if maintain_history:
                messages.extend(self.conversation_history)

            messages.append({"role": "user", "content": user_input})

            main_logger.debug(f"Querying model: {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            assistant_response = response.choices[0].message.content

            if maintain_history:
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": assistant_response})

            main_logger.debug("Query successful")
            return assistant_response
        except Exception as e:
            main_logger.error(f"An error occurred: {e}", exc_info=True)
            return f"An error occurred: {str(e)}"


class MultimodalChatbot(BaseChatbot):
    def query(self, system_prompt, user_input, image_path=None, maintain_history=False):
        try:
            messages = [{"role": "system", "content": system_prompt}]

            if maintain_history:
                messages.extend(self.conversation_history)

            if image_path:
                base64_image = self.encode_image(image_path)
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_input},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "auto"
                            }
                        }
                    ]
                }
            else:
                user_message = {"role": "user", "content": user_input}

            messages.append(user_message)

            main_logger.debug(f"Querying model: {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            assistant_response = response.choices[0].message.content

            if maintain_history:
                self.conversation_history.append(user_message)
                self.conversation_history.append({"role": "assistant", "content": assistant_response})

            main_logger.debug("Query successful")
            return assistant_response
        except Exception as e:
            main_logger.error(f"An error occurred: {e}", exc_info=True)
            return f"An error occurred: {str(e)}"

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


# 使用示例
if __name__ == "__main__":
    text_bot = TextChatbot(model_name="gpt-4o")
    multi_bot = MultimodalChatbot(model_name="gpt-4o")

    # 测试纯文本多轮对话
    system_prompt = "You are a helpful assistant."

    response1 = text_bot.query(system_prompt, user_input="What is the capital of France?", maintain_history=True)
    print("Response 1:", response1)

    response2 = text_bot.query(system_prompt, user_input="What is its population?",maintain_history=True) # maintain_history 默认为 False
    print("Response 2:", response2)

    # 清除历史
    text_bot.clear_history()

    # 测试多模态多轮对话
    image_path = r"D:\pythoncode\vlm_attention_starcraft2-main\vlm_attention\utils\9a64ffa2e5ae3562902565b7cf9bb34.png"
    starcraft_system_prompt = """

    """

    response3 = multi_bot.query(starcraft_system_prompt, user_input="Identify the most important enemy units.", image_path=image_path,maintain_history=True)
    print("Response 3:", response3)

    response4 = multi_bot.query(starcraft_system_prompt, user_input="What strategy would you recommend for our units?",maintain_history=True) # also support pure text
    print("Response 4:", response4)

    # 清除历史
    multi_bot.clear_history()
