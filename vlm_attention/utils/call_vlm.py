import base64
import logging
import os
from abc import ABC
from logging.handlers import RotatingFileHandler

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from openai import OpenAI
from PIL import Image

from vlm_attention.config.config import get_config


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


main_logger = setup_logger('main_logger', 'chatbot.log', level=logging.DEBUG)


class BaseChatbot(ABC):

    def __init__(
        self,
        model_name: str,
        use_camel: bool = False,
        chatgpt_kwargs: dict = None,
        agent_kwargs: dict = None,
        use_proxy: bool = False,
    ):
        if use_proxy:
            proxy_url = get_config("proxy", "url")
            os.environ["http_proxy"] = proxy_url
            os.environ["https_proxy"] = proxy_url
            main_logger.debug(f"Proxy set to: {proxy_url}")

        if use_camel:
            agent_kwargs = agent_kwargs or {}
            chatgpt_kwargs = chatgpt_kwargs or {}
            api_key = get_config("openai", "api_key")
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType(model_name),
                model_config_dict=ChatGPTConfig(**chatgpt_kwargs).as_dict(),
                api_key=api_key,
            )
            self.client = ChatAgent(model=model, **agent_kwargs)
        else:
            api_key = get_config("openai", "api_key")
            self.client = OpenAI(
                api_key=api_key,
                # organization=get_config("openai", "organization_id")
            )
        main_logger.debug("OpenAI client created")

        self.model = model_name
        self.use_camel = use_camel
        main_logger.debug(f"Model set to: {self.model}")
        self.conversation_history = []

    def query(self, system_prompt: str, user_input: str) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    def clear_history(self):
        if not self.use_camel:
            self.conversation_history = []
        else:
            self.client.memory.clear()


class TextChatbot(BaseChatbot):

    def query(
        self,
        system_prompt: str,
        user_input: str,
        maintain_history: bool = False,
    ) -> str:
        if not self.use_camel:
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
                    self.conversation_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    self.conversation_history.append({
                        "role":
                        "assistant",
                        "content":
                        assistant_response
                    })

                main_logger.debug("Query successful")
                return assistant_response
            except Exception as e:
                main_logger.error(f"An error occurred: {e}", exc_info=True)
                return f"An error occurred: {str(e)}"
        else:
            # Set the system message for the Camel agent
            self.client.system_message = system_prompt
            response = self.client.step(user_input)
            return response.msgs[0].content


class MultimodalChatbot(BaseChatbot):

    def query(
        self,
        system_prompt: str,
        user_input: str,
        image_path: str | None = None,
        maintain_history: bool = False,
    ) -> str:
        if not self.use_camel:
            try:
                messages = [{"role": "system", "content": system_prompt}]

                if maintain_history:
                    messages.extend(self.conversation_history)

                if image_path:
                    base64_image = self.encode_image(image_path)
                    user_message = {
                        "role":
                        "user",
                        "content": [{
                            "type": "text",
                            "text": user_input
                        }, {
                            "type": "image_url",
                            "image_url": {
                                "url":
                                f"data:image/jpeg;base64,{base64_image}",
                                "detail": "auto"
                            }
                        }]
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
                    self.conversation_history.append({
                        "role":
                        "assistant",
                        "content":
                        assistant_response
                    })

                main_logger.debug("Query successful")
                return assistant_response
            except Exception as e:
                main_logger.error(f"An error occurred: {e}", exc_info=True)
                return f"An error occurred: {str(e)}"
        else:
            self.client.system_message = system_prompt
            if image_path:
                image_list = [Image.open(image_path)]
            else:
                image_list = []
            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=user_input,
                image_list=image_list,
                image_detail="high",
            )
            response = self.client.step(user_msg)
            return response.msgs[0].content

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


if __name__ == "__main__":
    text_bot = TextChatbot(model_name="gpt-4o")
    text_bot_camel = TextChatbot(model_name="gpt-4o", use_camel=True)
    system_prompt = "You are a helpful assistant."
    response1 = text_bot.query(system_prompt,
                               user_input="What is the capital of France?",
                               maintain_history=True)
    print("Response 1:", response1)
    response2 = text_bot.query(system_prompt,
                               user_input="What is its population?",
                               maintain_history=True)
    print("Response 2:", response2)
    text_bot.clear_history()

    response1_camel = text_bot_camel.query(
        system_prompt, user_input="What is the capital of France?")
    print("Response 1 camel:", response1_camel)
    response2_camel = text_bot_camel.query(
        system_prompt, user_input="What is its population?")
    print("Response 2 camel:", response2_camel)
    text_bot_camel.clear_history()

    multi_bot = MultimodalChatbot(model_name="gpt-4o")
    multi_bot_camel = MultimodalChatbot(model_name="gpt-4o", use_camel=True)
    image_path = r"D:\pythoncode\vlm_attention_starcraft2-main\vlm_attention\utils\9a64ffa2e5ae3562902565b7cf9bb34.png"
    # image_path = "/Users/zecheng/code/vlm_attention_starcraft2/vlm_attention/utils/9a64ffa2e5ae3562902565b7cf9bb34.png"
    starcraft_system_prompt = "You are a helpful StartCraft 2 assistant."
    response3 = multi_bot.query(
        starcraft_system_prompt,
        user_input="Identify the most important enemy units.",
        image_path=image_path,
        maintain_history=True)
    print("Response 3:", response3)
    response4 = multi_bot.query(
        starcraft_system_prompt,
        user_input="What strategy would you recommend for our units?",
        maintain_history=True)
    print("Response 4:", response4)
    multi_bot.clear_history()

    response3_camel = multi_bot_camel.query(
        starcraft_system_prompt,
        user_input="Identify the most important enemy units.",
        image_path=image_path,
        maintain_history=True)
    print("Response 3 camel:", response3_camel)
    response4_camel = multi_bot_camel.query(
        starcraft_system_prompt,
        user_input="What strategy would you recommend for our units?",
        maintain_history=True)
    print("Response 4 camel:", response4_camel)
    multi_bot_camel.clear_history()
