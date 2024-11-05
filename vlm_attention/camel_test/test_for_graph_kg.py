import os
from typing import List
import json
import agentops

from camel.agents.chat_agent import FunctionCallingRecord
from camel.societies import RolePlaying
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, StorageType
from camel.configs import ChatGPTConfig
from camel.loaders import Firecrawl
from camel.retrievers import AutoRetriever
from camel.toolkits import OpenAIFunction, SearchToolkit
from camel.embeddings import MistralEmbedding

# Set up proxy
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# Set up API keys
os.environ["AGENTOPS_API_KEY"] = "8b18f0ae-4f83-4180-9866-a57d3267f063"
os.environ["OPENAI_API_BASE_URL"] = "https://ngedlktfticp.cloud.sealos.io/v1"
os.environ["OPENAI_API_KEY"] = "sk-rbXhBbN3c2outTJX5aBdF263457341F58eD18e5c325f18Be"
os.environ["FIRECRAWL_API_KEY"] = "fc-96cdb2de30954131b1a9e6cc8bb34d65"

# Initialize AgentOps
agentops.init(default_tags=["CAMEL cookbook"])

# Set up GPT-4 model
GPT4o_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
    model_config_dict=ChatGPTConfig(temperature=0.2).as_dict(),
)



# Define retrieval function
def retrieve_information_from_urls(urls: List[str], query: str) -> str:
    aggregated_content = ''
    for url in urls:
        scraped_content = Firecrawl().tidy_scrape(url)
        aggregated_content += scraped_content

    auto_retriever = AutoRetriever(
        vector_storage_local_path="local_data",
        storage_type=StorageType.QDRANT,
        embedding_model=MistralEmbedding(),
    )

    retrieved_info = auto_retriever.run_vector_retriever(
        query=query,
        contents=aggregated_content,
        top_k=3,
        similarity_threshold=0.5,
    )
    return retrieved_info

# Define knowledge graph builder function
def knowledge_graph_builder(text_input: str) -> str:
    print("Knowledge graph builder function called with input:", text_input)
    # In a real implementation, this function would create an actual knowledge graph
    # For this example, we'll return a simplified JSON representation
    graph = {
        "nodes": [
            {"id": "Turkish_shooter", "label": "Turkish Shooter"},
            {"id": "Paris_Olympics_2024", "label": "2024 Paris Olympics"},
            {"id": "Achievements", "label": "Achievements"},
        ],
        "edges": [
            {"source": "Turkish_shooter", "target": "Paris_Olympics_2024", "label": "participates in"},
            {"source": "Turkish_shooter", "target": "Achievements", "label": "has"},
        ]
    }
    return json.dumps(graph)

# Set up tools with descriptions
retrieval_tool = OpenAIFunction(retrieve_information_from_urls)
retrieval_tool.set_function_description("Retrieves relevant information from a list of URLs based on a given query.")
retrieval_tool.set_paramter_description("urls", "A list of URLs to retrieve information from")
retrieval_tool.set_paramter_description("query", "The query string to search for in the retrieved content")

search_tool = OpenAIFunction(SearchToolkit().search_duckduckgo)
search_tool.set_function_description("Performs a web search using DuckDuckGo and returns relevant URLs.")
search_tool.set_paramter_description("query", "The search query to be used")

knowledge_graph_tool = OpenAIFunction(knowledge_graph_builder)
knowledge_graph_tool.set_function_description("Builds a knowledge graph from the given text input.")
knowledge_graph_tool.set_paramter_description("text_input", "The text input to build the knowledge graph from")

tool_list = [retrieval_tool, search_tool, knowledge_graph_tool]

# Configure assistant model
assistant_model_config = ChatGPTConfig(
    tools=tool_list,
    temperature=0.0,
)

# Define task prompt
task_prompt = """Do a comprehensive study of the Turkish shooter in 2024 Paris
Olympics, write a report for me, then create a knowledge graph for the report.
You should use search tool to get related URLs first, then use retrieval tool
to get the retrieved content back by providing the list of URLs, finally
use tool to build the knowledge graph to finish the task.
No more other actions needed"""

# Set up role-playing session
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
        functions=[tool.to_dict() for tool in tool_list],
        function_call="auto"
    ),
    user_agent_kwargs=dict(model=GPT4o_model),
    task_prompt=task_prompt,
    with_task_specify=False,
)

# Start the interaction between agents
n = 0
input_msg = role_play_session.init_chat()
final_knowledge_graph = None
final_report = ""

while n < 20:  # Limit the chat to 20 turns
    n += 1
    assistant_response, user_response = role_play_session.step(input_msg)

    print(f"\n--- Turn {n} ---")
    print(f"Assistant: {assistant_response.msg.content[:100]}...")  # Print first 100 chars of response
    print(f"Function calls: {assistant_response.info.get('tool_calls', [])}")

    if assistant_response.terminated or user_response.terminated:
        print("Interaction terminated.")
        break

    # Check if knowledge graph tool was called
    for call in assistant_response.info.get('tool_calls', []):
        if call['function']['name'] == 'knowledge_graph_builder':
            print("Knowledge graph builder function was called!")
            final_knowledge_graph = json.loads(call['function']['output'])
            print(f"Generated knowledge graph: {final_knowledge_graph}")

    # Update the final report with the assistant's response
    final_report = assistant_response.msg.content

    if "CAMEL_TASK_DONE" in user_response.msg.content:
        print("Task completed by user.")
        break

    input_msg = assistant_response.msg

# End the AgentOps session
agentops.end_session("Success")

# Display the final report and knowledge graph
print("\n--- Final Report ---")
print(final_report)

print("\n--- Knowledge Graph ---")
if final_knowledge_graph:
    print(json.dumps(final_knowledge_graph, indent=2))
else:
    print("No knowledge graph was generated during the interaction.")
    print("Debug info:")
    print(f"Type of final_knowledge_graph: {type(final_knowledge_graph)}")
    print(f"Value of final_knowledge_graph: {final_knowledge_graph}")

# Visualize the knowledge graph
try:
    import networkx as nx
    import matplotlib.pyplot as plt

    if final_knowledge_graph:
        G = nx.Graph()
        for node in final_knowledge_graph['nodes']:
            G.add_node(node['id'], label=node['label'])
        for edge in final_knowledge_graph['edges']:
            G.add_edge(edge['source'], edge['target'], label=edge['label'])

        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, node_labels, font_size=8)

        plt.title("Knowledge Graph of Turkish Shooter in 2024 Paris Olympics")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("knowledge_graph.png")
        print("Knowledge graph visualization saved as 'knowledge_graph.png'")
    else:
        print("No knowledge graph available to visualize.")
except ImportError:
    print("NetworkX and/or matplotlib not installed. Unable to visualize the knowledge graph.")

print("\nTask completed.")