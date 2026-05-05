import streamlit as st
import tempfile
import os
import io

from dotenv import load_dotenv
load_dotenv()

# LangChain core
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import create_retriever_tool
from langchain.chat_models import init_chat_model

# Community
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

# LangGraph
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

# Tavily
from langchain_tavily import TavilySearch

# Image extraction
import fitz
from PIL import Image as PILImage

# Gemini
from google import genai

from typing import Annotated, Sequence
from typing_extensions import TypedDict

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

# ✅ validate env keys early
if not os.getenv("llm_api"):
    st.error("Missing GROQ API key")
    st.stop()

llm = init_chat_model(
    model="groq:openai/gpt-oss-120b",
    api_key=os.getenv("llm_api")
)

tavily_tool = TavilySearch(max_results=3)

# ═══════════════════════════════════════════════════════════════
# PDF TEXT
# ═══════════════════════════════════════════════════════════════

def extract_text(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)

# ═══════════════════════════════════════════════════════════════
# IMAGE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_images(file_path):
    images = []
    doc = fitz.open(file_path)

    for page_num in range(len(doc)):
        for img in doc[page_num].get_images(full=True):
            base = doc.extract_image(img[0])
            images.append({
                "page": page_num,
                "image_bytes": base["image"]
            })

    return images


def process_images(images):
    """SAFE version (won’t crash app)"""

    api_key = os.getenv("vm_api")
    if not api_key:
        return []

    client = genai.Client(api_key=api_key)

    os.makedirs("images", exist_ok=True)
    docs = []

    for i, img in enumerate(images):
        try:
            path = f"images/img_{i}.png"

            with open(path, "wb") as f:
                f.write(img["image_bytes"])

            pil_img = PILImage.open(io.BytesIO(img["image_bytes"]))

            res = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[pil_img, "Describe this image"]
            )

            text = res.text if res else "No description"

        except Exception as e:
            text = f"Image processing failed: {str(e)}"
            path = None

        docs.append(
            Document(
                page_content=text,
                metadata={"image_path": path}
            )
        )

    return docs

# ═══════════════════════════════════════════════════════════════
# VECTOR STORE
# ═══════════════════════════════════════════════════════════════

def create_store(docs, name):
    if not docs:
        return None

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("mistral_api")
    )

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=name
    )

# ═══════════════════════════════════════════════════════════════
# GRAPH
# ═══════════════════════════════════════════════════════════════

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: list(x)+list(y)]
    images: list


def build_graph(text_tool, image_tool=None):
    tools = [text_tool, tavily_tool]
    if image_tool:
        tools.append(image_tool)

    def agent(state):
        model = llm.bind_tools(tools)
        return {"messages": [model.invoke(state["messages"])]}

    def generate(state):
        q = state["messages"][0].content
        ctx = state["messages"][-1].content

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer briefly using context"),
            ("human", "Context:\n{context}\n\nQuestion:\n{q}")
        ])

        chain = prompt | llm | StrOutputParser()
        ans = chain.invoke({"context": ctx, "q": q})

        return {"messages": [AIMessage(content=ans)]}

    wf = StateGraph(State)
    wf.add_node("agent", agent)
    wf.add_node("retrieve", ToolNode(tools))
    wf.add_node("generate", generate)

    wf.add_edge(START, "agent")
    wf.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
    wf.add_edge("retrieve", "generate")
    wf.add_edge("generate", END)

    return wf.compile()

# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════

st.title("🤖 RAG Chatbot")

file = st.file_uploader("Upload PDF", type="pdf")

if file and st.button("Process"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        path = tmp.name

    with st.spinner("Reading PDF..."):
        text_docs = extract_text(path)

    st.write("Chunks:", len(text_docs))

    if len(text_docs) == 0:
        st.error("❌ PDF text extraction failed")
        st.stop()

    with st.spinner("Processing Images..."):
        imgs = extract_images(path)
        img_docs = process_images(imgs)

    with st.spinner("Building DB..."):
        text_db = create_store(text_docs, "text")
        img_db = create_store(img_docs, "img")

    # ✅ FIXED HERE
    text_tool = create_retriever_tool(
        text_db.as_retriever(),
        name="pdf_retriever",
        description="Retrieve information from uploaded PDF"
    )

    img_tool = None
    if img_db:
        img_tool = create_retriever_tool(
            img_db.as_retriever(),
            name="image_retriever",
            description="Retrieve image-related information from PDF"
        )

    st.session_state.graph = build_graph(text_tool, img_tool)
    st.success("✅ Ready!")

# CHAT
if "graph" in st.session_state:
    q = st.text_input("Ask")

    if q:
        res = st.session_state.graph.invoke({
            "messages": [HumanMessage(content=q)],
            "images": []
        })

        st.write(res["messages"][-1].content)
