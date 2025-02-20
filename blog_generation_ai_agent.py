import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm=ChatOpenAI(model="gpt-4o")

def title_creator(topic:str) -> str:
    
    """
    Generates a catchy and relevant title for a blog post based on the given topic.
    
    Args:
        topic (str): The topic of the blog post.
    
    Returns:
        str: The generated title.
    """

    llm=ChatOpenAI(model="gpt-4o")
    
    messages1 = [
    (
        "system",
        "You are a helpful assistant that creates a catchy and relevant titles for a blog post, dont ask return question just answer it",
    ),
    ("human", topic),
]
    
    title=llm.invoke(messages1)
    
    return title.content


def generate_content(title:str)->str:
    
    """
    Generates a detailed and engaging blog post based on the given title and topic.
    
    Args:
        title (str): The title of the blog post.
    
    Returns:
        str: The generated blog content
    """
    
    llm=ChatOpenAI(model="gpt-4o")
    
    messages2 = [
    (
        "system",
        "You are a helpful assistant that writes a detailed and engaging blog post based on the topic and title, dont ask return question just answer it",
    ),
    ("human", title),
]
    
    
    content=llm.invoke(messages2)
    
    return content.content

tools=[title_creator,generate_content]

llm_with_tools=llm.bind_tools(tools,parallel_tool_calls=False)

sys_msg=SystemMessage(content="you are an helpful assistant tasked with writing the blog from the input")

def assistant(state:MessagesState):
    return {"messages":[llm_with_tools.invoke([sys_msg]+state["messages"])]}


builder=StateGraph(MessagesState)

builder.add_node("assistant",assistant)
builder.add_node("tools",ToolNode(tools))

builder.add_edge(START,"assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition
)

builder.add_edge("tools","assistant")

memory=MemorySaver()

react_graph=builder.compile(checkpointer=memory)

display(Image(react_graph.get_graph().draw_mermaid_png()))

config={"configurable":{"thread_id":"1"}}

messages=[HumanMessage(content="How AI is Transforming Education for Educators")]

messages = react_graph.invoke({"messages": messages},config)
for m in messages['messages']:
    m.pretty_print()