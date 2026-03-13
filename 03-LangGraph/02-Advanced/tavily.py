# %% [markdown]
# # Build a Travily and Invoke Gen AI with Bedrock Chatbot

# %%
# %%capture --no-stderr
# %pip install --quiet -U langchain_openai langchain_core langgraph langgraph-prebuilt langchain-tavily
# %pip install boto3
# %pip install --upgrade python-dotenv

# %%
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ["AWS_DEFAULT_REGION"] =os.getenv('AWS_DEFAULT_REGION')
os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# %%
os.environ["LANGSMITH_API_KEY"] = os.getenv('LANGSMITH_API_KEY')
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "tutorial"

# %% [markdown]
# ## Define the tool
# 
# Define the web search tool:
# 
# 

# %%
from langchain_tavily import TavilySearch

search_tool = TavilySearch(max_results=2)

# %% [markdown]
# # Define the tool use AWS API Gateway endpoint

# %%
from langchain_core.tools import tool

tools = [search_tool]
search_tool.invoke("What's a 'node' in LangGraph?")

# %% [markdown]
# ## Define the graph
# 
# 

# %%
from langchain_aws import ChatBedrock

llm = ChatBedrock(
   model_id="amazon.nova-lite-v1:0",
   temperature=1,
)

# %% [markdown]
# We can now incorporate it into a StateGraph:

# %%
from typing import Annotated

from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

# %%
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st

st.title("AWS Bedrock Agent Chat")
st.write("Chat with your AWS Bedrock Agent!")
user_input = st.text_input("user: ", "") 
if st.button("Send"):
    if user_input:
        with st.spinner("Waiting for agent response..."):
            response = graph.invoke({"messages": [HumanMessage(content=user_input)]})
        st.success("Agent response received!")
        st.write(f"agent: {response}")
    else:
        st.warning("Please enter a message to send to the agent.")




# %%
#print(result)


