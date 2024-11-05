import os
from getpass import getpass
from typing import List, Dict,Any
import json
import networkx as nx
import matplotlib.pyplot as plt

# 环境变量设置（保持不变）
os.environ["AGENTOPS_API_KEY"] = "8b18f0ae-4f83-4180-9866-a57d3267f063"
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["OPENAI_API_KEY"] = "sk-hiP_PDdVZTqFkXRZ6YE2FDRHojCl98uO2bxZboHul2T3BlbkFJ5IKXSHi1721jZ8buILxO0kP0KFpLnqyk99wlhqG_gA"
os.environ["OPENAI_ORG_ID"] ="org-YBLeiIenLZxY3ncCUDNLT67O"
os.environ["FIRECRAWL_API_KEY"] = "fc-96cdb2de30954131b1a9e6cc8bb34d65"

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig
from camel.loaders import Firecrawl
from camel.retrievers import AutoRetriever
from camel.toolkits import OpenAIFunction, SearchToolkit
from camel.types import StorageType
from camel.embeddings import OpenAIEmbedding
from camel.storages import Neo4jGraph
from camel.loaders import UnstructuredIO
from camel.agents import KnowledgeGraphAgent
from camel.societies import RolePlaying
from camel.agents.chat_agent import FunctionCallingRecord
from camel.utils import print_text_animated
import agentops
from colorama import Fore

# 设置模型
llm = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
    model_config_dict=ChatGPTConfig(temperature=0.2).as_dict(),
)


def visualize_knowledge_graph(graph_data):
    if not graph_data['nodes'] and not graph_data['relationships']:
        print("The graph is empty. No visualization created.")
        return None

    G = nx.Graph()

    # Add nodes
    for node in graph_data['nodes']:
        G.add_node(node['id'], label=node.get('properties', {}).get('name', node['id']))

    # Add edges
    for edge in graph_data['relationships']:
        G.add_edge(edge['subj']['id'], edge['obj']['id'], label=edge['type'])

    # Create visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8)

    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Knowledge Graph Visualization")
    plt.axis('off')
    plt.tight_layout()

    # Save image
    plt.savefig('knowledge_graph.png')
    plt.close()

    return 'knowledge_graph.png'

def retrieve_information_from_urls(urls: list[str], query: str) -> str:
    """
    Retrieves relevant information from a list of URLs based on a given query.

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
    for url in urls:
        scraped_content = Firecrawl().tidy_scrape(url)
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

def knowledge_graph_builder(text_input: str) -> Dict[str, Any]:
    n4j = Neo4jGraph(
          url="neo4j+s://20f8f7a0.databases.neo4j.io",
          username="neo4j",
          password="R_vYylPoHXsWTBG1GA_TGoeTQQmBgHCE39CnUUgjC5w",
    )
    uio = UnstructuredIO()
    kg_agent = KnowledgeGraphAgent(model=llm)

    element_example = uio.create_element_from_text(text_input, element_id="001")
    graph_elements = kg_agent.run(element_example, parse_graph_elements=True)

    if not isinstance(graph_elements, list):
        graph_elements = [graph_elements]

    n4j.add_graph_elements(graph_elements=graph_elements)

    # Extract nodes and relationships
    nodes = []
    edges = []

    for element in graph_elements:
        if isinstance(element, dict):
            if 'nodes' in element and 'relationships' in element:
                nodes.extend(element['nodes'])
                edges.extend(element['relationships'])
            elif element.get('type') == 'node':
                nodes.append(element)
            elif element.get('type') == 'relationship':
                edges.append(element)

    # Create simple text summary
    summary = {
        "nodes": [f"{node.get('properties', {}).get('name', node['id'])} (ID: {node['id']}, Type: {node['type']})" for node in nodes],
        "relationships": [f"{edge['subj']['id']} -{edge['type']}-> {edge['obj']['id']}" for edge in edges]
    }

    return {
        "graph_elements": {"nodes": nodes, "relationships": edges},
        "summary": summary
    }


task_prompt = """Do a comprehensive study of the Turkish shooter in 2024 paris
olympics, write a report for me, then create a knowledge graph for the report.
You should use search tool to get related URLs first, then use retrieval tool
to get the retrieved content back by providing the list of URLs, finially
use tool to build the knowledge graph to finish the task.
No more other actions needed"""

retrieval_tool = OpenAIFunction(
    func=retrieve_information_from_urls,
    openai_tool_schema={
        "type": "function",
        "function": {
            "name": "retrieve_information_from_urls",
            "description": "Retrieves relevant information from a list of URLs based on a given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of URLs to scrape content from."
                    },
                    "query": {
                        "type": "string",
                        "description": "The query string to search for relevant information."
                    }
                },
                "required": ["urls", "query"]
            }
        }
    }
)
search_tool = OpenAIFunction(SearchToolkit().search_duckduckgo)

knowledge_graph_tool = OpenAIFunction(
    func=knowledge_graph_builder,
    openai_tool_schema={
        "type": "function",
        "function": {
            "name": "knowledge_graph_builder",
            "description": "Build and store a knowledge graph from the provided text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_input": {
                        "type": "string",
                        "description": "The input text from which the knowledge graph is to be constructed."
                    }
                },
                "required": ["text_input"]
            }
        }
    }
)

tool_list = [
    retrieval_tool,
    search_tool,
    knowledge_graph_tool,
]

assistant_model_config = ChatGPTConfig(
    tools=tool_list,
    temperature=0.0,
)

# 初始化角色扮演会话
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

print(Fore.GREEN + f"AI Assistant sys message:\n{role_play_session.assistant_sys_msg}\n")
print(Fore.BLUE + f"AI User sys message:\n{role_play_session.user_sys_msg}\n")
print(Fore.YELLOW + f"Original task prompt:\n{task_prompt}\n")
print(Fore.CYAN + f"Specified task prompt:\n{role_play_session.specified_task_prompt}\n")
print(Fore.RED + f"Final task prompt:\n{role_play_session.task_prompt}\n")

agentops.init(default_tags=["CAMEL cookbook"])

n = 0
input_msg = role_play_session.init_chat()
final_report = ""
kg_result = None
while n < 20:
    n += 1
    assistant_response, user_response = role_play_session.step(input_msg)

    if assistant_response.terminated or user_response.terminated:
        print(Fore.GREEN + "Chat terminated.")
        break

    print_text_animated(Fore.BLUE + f"AI User:\n\n{user_response.msg.content}\n", 0.01)

    if "CAMEL_TASK_DONE" in user_response.msg.content:
        final_report = user_response.msg.content
        break

    print_text_animated(Fore.GREEN + "AI Assistant:", 0.01)
    tool_calls: List[FunctionCallingRecord] = [
        FunctionCallingRecord(**call.as_dict())
        for call in assistant_response.info['tool_calls']
    ]
    for func_record in tool_calls:
        print_text_animated(f"{func_record}", 0.01)
        # 检查 func_record 的结构并找到正确的属性来识别知识图谱构建函数
        if hasattr(func_record, 'function') and func_record.function.get('name') == "knowledge_graph_builder":
            kg_result = func_record.output
    print_text_animated(f"{assistant_response.msg.content}\n", 0.01)

    input_msg = assistant_response.msg

# 输出知识图谱结果
if kg_result:
    print("\nKnowledge Graph Summary:")
    print(json.dumps(kg_result['summary'], indent=2))

    # 使用可视化函数
    visualization_path = visualize_knowledge_graph(kg_result['graph_elements'])
    if visualization_path:
        print(f"\nKnowledge Graph visualization saved as: {visualization_path}")
    else:
        print("\nNo visualization created due to empty graph.")

    # 打印详细的图元素信息
    print("\nDetailed Graph Elements:")
    print(json.dumps(kg_result['graph_elements'], indent=2))
elif final_report:
    try:
        kg_result = knowledge_graph_tool.func(text_input=final_report)
        print("\nKnowledge Graph Summary:")
        print(json.dumps(kg_result['summary'], indent=2))

        # 使用可视化函数
        visualization_path = visualize_knowledge_graph(kg_result['graph_elements'])
        if visualization_path:
            print(f"\nKnowledge Graph visualization saved as: {visualization_path}")
        else:
            print("\nNo visualization created due to empty graph.")

        # 打印详细的图元素信息
        print("\nDetailed Graph Elements:")
        print(json.dumps(kg_result['graph_elements'], indent=2))
    except Exception as e:
        print(f"Error generating or visualizing knowledge graph: {str(e)}")
else:
    print("No knowledge graph or final report generated.")

agentops.end_session("Success")