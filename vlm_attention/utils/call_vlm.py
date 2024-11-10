import logging
import os
from abc import ABC
from logging.handlers import RotatingFileHandler
from typing import Optional, List

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.responses import ChatAgentResponse
from camel.types import ModelType, RoleType, ModelPlatformType
from camel.models import ModelFactory, BaseModelBackend
from camel.configs import ChatGPTConfig
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
            is_multimodal: bool = False,
            system_prompt: str = "You are a helpful assistant.",
            chatgpt_kwargs: Optional[dict] = None,
            token_limit: Optional[int] = None,
            use_proxy: bool = False,
    ):
        if use_proxy:
            proxy_url = get_config("proxy", "url")
            os.environ["http_proxy"] = proxy_url
            os.environ["https_proxy"] = proxy_url
            main_logger.debug(f"Proxy set to: {proxy_url}")

        # Get configuration values
        api_key = get_config("openai", "api_key")
        organization_id = get_config("openai", "organization_id")

        # Select appropriate model based on type
        model_name = (
            get_config("openai", "vlm_model_name") if is_multimodal
            else get_config("openai", "llm_model_name")
        )

        # Create base message for system prompt
        self.system_message = BaseMessage(
            role_name="Assistant",
            role_type=RoleType.ASSISTANT,
            meta_dict={"organization_id": organization_id},
            content=system_prompt
        )

        # Initialize ChatGPT configuration
        config_dict = ChatGPTConfig(**(chatgpt_kwargs or {})).as_dict()

        # Create model backend
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType(model_name),
            model_config_dict=config_dict,
            api_key=api_key,
        )

        # Create chat agent
        self.chat_agent = ChatAgent(
            system_message=self.system_message,
            model=model,
            token_limit=token_limit,
        )

        main_logger.debug(f"Model set to: {model_name}")

    def query(self, user_input: str) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    def clear_history(self):
        self.chat_agent.reset()


class TextChatbot(BaseChatbot):
    def __init__(
            self,
            system_prompt: str = "You are a helpful assistant.",
            chatgpt_kwargs: Optional[dict] = None,
            token_limit: Optional[int] = None,
            use_proxy: bool = False,
    ):
        super().__init__(
            is_multimodal=False,
            system_prompt=system_prompt,
            chatgpt_kwargs=chatgpt_kwargs,
            token_limit=token_limit,
            use_proxy=use_proxy,
        )

    def query(self, user_input: str) -> str:
        try:
            # Create user message
            user_message = BaseMessage(
                role_name="User",
                role_type=RoleType.USER,
                meta_dict={},
                content=user_input
            )

            # Get response from chat agent
            response: ChatAgentResponse = self.chat_agent.step(user_message)

            if response.msgs:
                main_logger.debug("Query successful")
                return response.msgs[0].content
            else:
                return "No response generated"

        except Exception as e:
            main_logger.error(f"An error occurred: {e}", exc_info=True)
            return f"An error occurred: {str(e)}"


class MultimodalChatbot(BaseChatbot):
    def __init__(
            self,
            system_prompt: str = "You are a helpful vision assistant.",
            chatgpt_kwargs: Optional[dict] = None,
            token_limit: Optional[int] = None,
            use_proxy: bool = False,
    ):
        super().__init__(
            is_multimodal=True,
            system_prompt=system_prompt,
            chatgpt_kwargs=chatgpt_kwargs,
            token_limit=token_limit,
            use_proxy=use_proxy,
        )

    def query(
            self,
            user_input: str,
            image_path: Optional[str] = None,
    ) -> str:
        try:
            # Process image if provided
            image_list = [Image.open(image_path)] if image_path else []

            # Create user message with image if available
            user_message = BaseMessage(
                role_name="User",
                role_type=RoleType.USER,
                meta_dict={},
                content=user_input,
                image_list=image_list,
                image_detail="high"
            )

            # Get response from chat agent
            response: ChatAgentResponse = self.chat_agent.step(user_message)

            if response.msgs:
                main_logger.debug("Query successful")
                return response.msgs[0].content
            else:
                return "No response generated"

        except Exception as e:
            main_logger.error(f"An error occurred: {e}", exc_info=True)
            return f"An error occurred: {str(e)}"


def test_chat_history():
    # 1. 创建文本聊天机器人
    text_bot = TextChatbot(
        system_prompt="You are a helpful assistant.",
        use_proxy=True
    )

    # 2. 第一轮对话 - 设定上下文
    response1 = text_bot.query("Let's talk about Paris. What's the capital city of France?")
    print("\nQuestion 1: Let's talk about Paris. What's the capital city of France?")
    print("Response 1:", response1)

    # 3. 第二轮对话 - 使用代词,测试是否理解上下文
    response2 = text_bot.query("What is its population?")
    print("\nQuestion 2: What is its population?")
    print("Response 2:", response2)

    # 4. 第三轮对话 - 继续使用上下文
    response3 = text_bot.query("And what are some famous landmarks in this city?")
    print("\nQuestion 3: And what are some famous landmarks in this city?")
    print("Response 3:", response3)

    # 5. 清除历史后测试
    text_bot.clear_history()

    # 6. 使用相同的代词提问,但应该无法得到正确回答因为没有上下文
    response4 = text_bot.query("What is its population?")
    print("\nAfter clearing history - Question: What is its population?")
    print("Response 4:", response4)


def test_camel():
    # Test the text chatbot
    text_bot = TextChatbot(
        system_prompt="You are a helpful assistant.",
        use_proxy=True
    )

    response1 = text_bot.query("What is the capital of France?")
    print("Response 1:", response1)
    response2 = text_bot.query("What is its population?")
    print("Response 2:", response2)
    text_bot.clear_history()

    # Test the multimodal chatbot
    starcraft_system_prompt = "You are a helpful StarCraft 2 assistant."
    multi_bot = MultimodalChatbot(
        system_prompt=starcraft_system_prompt,
        use_proxy=True
    )

    image_path = r"D:\pythoncode\vlm_attention_starcraft2-main\vlm_attention\utils\9a64ffa2e5ae3562902565b7cf9bb34.png"
    response3 = multi_bot.query(
        user_input="Identify the most important enemy units.",
        image_path=image_path
    )
    print("Response 3:", response3)
    response4 = multi_bot.query(
        user_input="What strategy would you recommend for our units?"
    )
    print("Response 4:", response4)
    multi_bot.clear_history()

if __name__ == "__main__":
    test_chat_history()