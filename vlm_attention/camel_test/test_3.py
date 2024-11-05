import os
from getpass import getpass

# Prompt for the AgentOps API key securely
os.environ["AGENTOPS_API_KEY"] = "8b18f0ae-4f83-4180-9866-a57d3267f063"
# Prompt for the API key securely
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["OPENAI_API_KEY"] = "sk-hiP_PDdVZTqFkXRZ6YE2FDRHojCl98uO2bxZboHul2T3BlbkFJ5IKXSHi1721jZ8buILxO0kP0KFpLnqyk99wlhqG_gA"
os.environ["OPENAI_ORG_ID"] ="org-YBLeiIenLZxY3ncCUDNLT67O"
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig

# Set up model
llm = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
    model_config_dict=ChatGPTConfig(temperature=0.2).as_dict(),
)
# Prompt for the Firecrawl API key securely
os.environ["FIRECRAWL_API_KEY"] = "fc-96cdb2de30954131b1a9e6cc8bb34d65"
from camel.loaders import Firecrawl

firecrawl = Firecrawl()

# Scrape and clean content from a specified URL
response = firecrawl.tidy_scrape(
    url="https://www.camel-ai.org/post/crab"
)

print(response)
from camel.retrievers import AutoRetriever
from camel.toolkits import OpenAIFunction, SearchToolkit
from camel.types import ModelPlatformType, ModelType, StorageType
from camel.embeddings import OpenAIEmbedding
def retrieve_information_from_urls(urls: list[str], query: str) -> str:
    r"""Retrieves relevant information from a list of URLs based on a given
    query.

    This function uses the `Firecrawl` tool to scrape content from the
    provided URLs and then uses the `AutoRetriever` from CAMEL to retrieve the
    most relevant information based on the query from the scraped content.

    Args:
        urls (list[str]): A list of URLs to scrape content from.
        query (str): The query string to search for relevant information.

    Returns:
        str: The most relevant information retrieved based on the query.
    """
    aggregated_content = ''

    # Scrape and aggregate content from each URL
    for url in urls:
        scraped_content = Firecrawl().tidy_scrape(url)
        aggregated_content += scraped_content

    # Set up a vector retriever with local storage and embedding model from Mistral AI
    auto_retriever = AutoRetriever(
        vector_storage_local_path="local_data",
        storage_type=StorageType.QDRANT,
        embedding_model=OpenAIEmbedding(),
    )

    # Retrieve the most relevant information based on the query
    # You can adjust the top_k and similarity_threshold value based on your needs
    retrieved_info = auto_retriever.run_vector_retriever(
        query=query,
        contents=aggregated_content,
        top_k=3,
        similarity_threshold=0.5,
    )

    return retrieved_info

retrieved_info = retrieve_information_from_urls(
    query="Which country won the most golden prize in 2024 Olympics?",
    urls=[
        "https://www.nbcnews.com/sports/olympics/united-states-china-gold-medals-rcna166013",
    ],
)

print(retrieved_info)
import agentops
agentops.init(default_tags=["CAMEL cookbook"])
from camel.storages import Neo4jGraph
from camel.loaders import UnstructuredIO
from camel.agents import KnowledgeGraphAgent

def knowledge_graph_builder(text_input: str) -> None:
    r"""Build and store a knowledge graph from the provided text.

    This function processes the input text to create and extract nodes and relationships,
    which are then added to a Neo4j database as a knowledge graph.

    Args:
        text_input (str): The input text from which the knowledge graph is to be constructed.

    Returns:
        graph_elements: The generated graph element from knowlegde graph agent.
    """

    # Set Neo4j instance
    n4j = Neo4jGraph(
          url="neo4j+s://20f8f7a0.databases.neo4j.io",
          username="neo4j",
          password="R_vYylPoHXsWTBG1GA_TGoeTQQmBgHCE39CnUUgjC5w",
            )
    # Initialize instances
    uio = UnstructuredIO()
    kg_agent = KnowledgeGraphAgent(model=llm)

    # Create an element from the provided text
    element_example = uio.create_element_from_text(text_input, element_id="001")

    # Extract nodes and relationships using the Knowledge Graph Agent
    graph_elements = kg_agent.run(element_example, parse_graph_elements=True)

    # Add the extracted graph elements to the Neo4j database
    n4j.add_graph_elements(graph_elements=[graph_elements])

    return graph_elements
from typing import List

from colorama import Fore

from camel.agents.chat_agent import FunctionCallingRecord
from camel.societies import RolePlaying
from camel.utils import print_text_animated
task_prompt = """Do a comprehensive study of the Turkish shooter in 2024 paris
olympics, write a report for me, then create a knowledge graph for the report.
You should use search tool to get related URLs first, then use retrieval tool
to get the retrieved content back by providing the list of URLs, finially
use tool to build the knowledge graph to finish the task.
No more other actions needed"""
retrieval_tool = OpenAIFunction(retrieve_information_from_urls)
search_tool = OpenAIFunction(SearchToolkit().search_duckduckgo)
knowledge_graph_tool = OpenAIFunction(knowledge_graph_builder)

tool_list = [
    retrieval_tool,
    search_tool,
    knowledge_graph_tool,
]

assistant_model_config = ChatGPTConfig(
    tools=tool_list,
    temperature=0.0,
)
# Initialize the role-playing session
role_play_session = RolePlaying(
    assistant_role_name="CAMEL Assistant",
    user_role_name="CAMEL User",
    assistant_agent_kwargs=dict(
        model=ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=assistant_model_config.as_dict(),
        ),
        tools=tool_list,
    ),
    user_agent_kwargs=dict(model=llm),
    task_prompt=task_prompt,
    with_task_specify=False,
)
# Print system and task messages
print(
    Fore.GREEN
    + f"AI Assistant sys message:\n{role_play_session.assistant_sys_msg}\n"
)
print(Fore.BLUE + f"AI User sys message:\n{role_play_session.user_sys_msg}\n")

print(Fore.YELLOW + f"Original task prompt:\n{task_prompt}\n")
print(
    Fore.CYAN
    + "Specified task prompt:"
    + f"\n{role_play_session.specified_task_prompt}\n"
)
print(Fore.RED + f"Final task prompt:\n{role_play_session.task_prompt}\n")
n = 0
input_msg = role_play_session.init_chat()
while n < 20: # Limit the chat to 20 turns
    n += 1
    assistant_response, user_response = role_play_session.step(input_msg)

    if assistant_response.terminated:
        print(
            Fore.GREEN
            + (
                "AI Assistant terminated. Reason: "
                f"{assistant_response.info['termination_reasons']}."
            )
        )
        break
    if user_response.terminated:
        print(
            Fore.GREEN
            + (
                "AI User terminated. "
                f"Reason: {user_response.info['termination_reasons']}."
            )
        )
        break
    # Print output from the user
    print_text_animated(
        Fore.BLUE + f"AI User:\n\n{user_response.msg.content}\n",
        0.01
    )

    if "CAMEL_TASK_DONE" in user_response.msg.content:
        break

    # Print output from the assistant, including any function
    # execution information
    print_text_animated(Fore.GREEN + "AI Assistant:", 0.01)
    tool_calls: List[FunctionCallingRecord] = [
        FunctionCallingRecord(**call.as_dict())
        for call in assistant_response.info['tool_calls']
    ]
    for func_record in tool_calls:
        print_text_animated(f"{func_record}", 0.01)
    print_text_animated(f"{assistant_response.msg.content}\n", 0.01)

    input_msg = assistant_response.msg
    # End the AgentOps session
    agentops.end_session("Success")