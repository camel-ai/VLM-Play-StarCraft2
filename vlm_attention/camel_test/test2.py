import os
import json
from typing import List, Dict, Any
from colorama import Fore

# Import CAMEL modules
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, StorageType
from camel.configs import ChatGPTConfig
from camel.loaders import Firecrawl, UnstructuredIO
from camel.retrievers import AutoRetriever
from camel.toolkits import OpenAIFunction, SearchToolkit
from camel.embeddings import OpenAIEmbedding
from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.societies import RolePlaying
from camel.agents.chat_agent import FunctionCallingRecord
from camel.utils import print_text_animated

# Import AgentOps
import agentops


class CAMELDemo:
    def __init__(self):
        self.setup_model()
        self.setup_tools()
        self.kg_result = None



    def setup_model(self):
        self.GPT4O = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig(temperature=0.2).as_dict(),
        )

    def setup_tools(self):
        self.firecrawl = Firecrawl()

        # Retrieve information from URLs tool
        retrieve_schema = {
            "type": "function",
            "function": {
                "name": "retrieve_information_from_urls",
                "description": "Retrieve relevant information from a list of URLs based on a given query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of URLs to retrieve information from"
                        },
                        "query": {
                            "type": "string",
                            "description": "The query to use for retrieving relevant information"
                        }
                    },
                    "required": ["urls", "query"]
                }
            }
        }
        self.retrieval_tool = OpenAIFunction(self.retrieve_information_from_urls, retrieve_schema)

        # Search tool
        search_schema = {
            "type": "function",
            "function": {
                "name": "search_duckduckgo",
                "description": "Search the web using DuckDuckGo and return relevant results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        self.search_tool = OpenAIFunction(SearchToolkit().search_duckduckgo, search_schema)

        # Knowledge graph builder tool
        kg_schema = {
            "type": "function",
            "function": {
                "name": "knowledge_graph_builder",
                "description": "Build a knowledge graph from the provided text input.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text_input": {
                            "type": "string",
                            "description": "The text input to build a knowledge graph from"
                        }
                    },
                    "required": ["text_input"]
                }
            }
        }
        self.knowledge_graph_tool = OpenAIFunction(self.knowledge_graph_builder, kg_schema)

        self.tool_list = [self.retrieval_tool, self.search_tool, self.knowledge_graph_tool]

    def retrieve_information_from_urls(self, urls: List[str], query: str) -> str:
        aggregated_content = ''
        for url in urls:
            scraped_content = self.firecrawl.tidy_scrape(url)
            aggregated_content += scraped_content

        auto_retriever = AutoRetriever(
            vector_storage_local_path="local_data",
            storage_type=StorageType.QDRANT,
            embedding_model=OpenAIEmbedding(),
        )

        retrieved_info = auto_retriever.run_vector_retriever(
            query=query,
            contents=aggregated_content,
            top_k=3,
            similarity_threshold=0.5,
        )

        return retrieved_info

    def knowledge_graph_builder(self, text_input: str) -> Dict[str, Any]:
        try:
            n4j = Neo4jGraph(
                url="neo4j+s://20f8f7a0.databases.neo4j.io",
                username="neo4j",
                password="R_vYylPoHXsWTBG1GA_TGoeTQQmBgHCE39CnUUgjC5w",
            )

            uio = UnstructuredIO()
            kg_agent = KnowledgeGraphAgent(model=self.GPT4O)

            element_example = uio.create_element_from_text(text_input, element_id="001")
            graph_elements = kg_agent.run(element_example, parse_graph_elements=True)
            n4j.add_graph_elements(graph_elements=[graph_elements])

            # Store the result for later saving
            self.kg_result = graph_elements

            print(f"Knowledge graph created with {len(graph_elements)} elements.")
            return graph_elements
        except Exception as e:
            print(f"Error in knowledge_graph_builder: {str(e)}")
            return {}
    def save_knowledge_graph(self, filename: str = "knowledge_graph.json"):
        if self.kg_result:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.kg_result, f, indent=2)
                print(f"Knowledge graph saved to {filename}")
            except Exception as e:
                print(f"Error saving knowledge graph: {str(e)}")
        else:
            print("No knowledge graph knowledge_data to save.")

    def run_role_playing_session(self):
        task_prompt = """Do a comprehensive study of the Turkish shooter in 2024 paris
        olympics, write a report for me, then create a knowledge graph for the report.
        You should use search tool to get related URLs first, then use retrieval tool
        to get the retrieved content back by providing the list of URLs, finially
        use tool to build the knowledge graph to finish the task.
        No more other actions needed"""

        assistant_model_config = ChatGPTConfig(
            tools=self.tool_list,
            temperature=0.0,
        )

        role_play_session = RolePlaying(
            assistant_role_name="CAMEL Assistant",
            user_role_name="CAMEL User",
            assistant_agent_kwargs=dict(
                model=ModelFactory.create(
                    model_platform=ModelPlatformType.OPENAI,
                    model_type=ModelType.GPT_4,
                    model_config_dict=assistant_model_config.as_dict(),
                ),
                tools=self.tool_list,
            ),
            user_agent_kwargs=dict(model=self.GPT4O),
            task_prompt=task_prompt,
            with_task_specify=False,
        )

        self.print_session_info(role_play_session)

        n = 0
        input_msg = role_play_session.init_chat()
        while n < 20:
            n += 1
            assistant_response, user_response = role_play_session.step(input_msg)

            if assistant_response.terminated or user_response.terminated:
                self.print_termination_reason(assistant_response, user_response)
                break

            self.print_user_response(user_response)
            if "CAMEL_TASK_DONE" in user_response.msg.content:
                break

            self.print_assistant_response(assistant_response)
            input_msg = assistant_response.msg

        # Save the knowledge graph after the session
        self.save_knowledge_graph()

    def print_session_info(self, role_play_session):
        print(Fore.GREEN + f"AI Assistant sys message:\n{role_play_session.assistant_sys_msg}\n")
        print(Fore.BLUE + f"AI User sys message:\n{role_play_session.user_sys_msg}\n")
        print(Fore.YELLOW + f"Original task prompt:\n{role_play_session.task_prompt}\n")
        print(Fore.CYAN + f"Specified task prompt:\n{role_play_session.specified_task_prompt}\n")
        print(Fore.RED + f"Final task prompt:\n{role_play_session.task_prompt}\n")

    def print_termination_reason(self, assistant_response, user_response):
        if assistant_response.terminated:
            print(Fore.GREEN + f"AI Assistant terminated. Reason: {assistant_response.info['termination_reasons']}.")
        if user_response.terminated:
            print(Fore.GREEN + f"AI User terminated. Reason: {user_response.info['termination_reasons']}.")

    def print_user_response(self, user_response):
        print_text_animated(Fore.BLUE + f"AI User:\n\n{user_response.msg.content}\n", 0.01)

    def print_assistant_response(self, assistant_response):
        print_text_animated(Fore.GREEN + "AI Assistant:", 0.01)
        tool_calls: List[FunctionCallingRecord] = [
            FunctionCallingRecord(**call.as_dict())
            for call in assistant_response.info['tool_calls']
        ]
        for func_record in tool_calls:
            print_text_animated(f"{func_record}", 0.01)
        print_text_animated(f"{assistant_response.msg.content}\n", 0.01)


if __name__ == "__main__":
    # Set up proxy
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

    # Set up API keys
    os.environ["AGENTOPS_API_KEY"] = "8b18f0ae-4f83-4180-9866-a57d3267f063"

    os.environ[
        "OPENAI_API_KEY"] = "sk-hiP_PDdVZTqFkXRZ6YE2FDRHojCl98uO2bxZboHul2T3BlbkFJ5IKXSHi1721jZ8buILxO0kP0KFpLnqyk99wlhqG_gA"
    os.environ["OPENAI_ORG_ID"] = "org-YBLeiIenLZxY3ncCUDNLT67O"
    os.environ["FIRECRAWL_API_KEY"] = "fc-96cdb2de30954131b1a9e6cc8bb34d65"
    try:
        # Initialize AgentOps session
        agentops_session = agentops.init(default_tags=["CAMEL cookbook"])
        print("AgentOps session initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize AgentOps session: {str(e)}")
        agentops_session = None

    demo = CAMELDemo()
    demo.run_role_playing_session()

    # End AgentOps session
    if agentops_session:
        try:
            agentops.end_session("Success")
            print("AgentOps session ended successfully.")
        except Exception as e:
            print(f"Failed to end AgentOps session: {str(e)}")
    else:
        print("Warning: AgentOps session was not initialized.")

    # End AgentOps session
    if agentops_session:
        agentops.end_session("Success")
    else:
        print("Warning: AgentOps session was not initialized.")