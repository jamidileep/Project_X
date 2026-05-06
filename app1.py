import streamlit as st
import tempfile
import os
import io

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import create_retriever_tool
from langchain.chat_models import init_chat_model

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_tavily import TavilySearch

import fitz
from PIL import Image as PILImage
from google import genai

from typing import Annotated, Sequence
from typing_extensions import TypedDict

# ════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════

llm = init_chat_model(
    model="groq:openai/gpt-oss-120b",
    api_key=os.getenv("llm_api")
)

tavily_tool = TavilySearch(max_results=3)

# ════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(docs)

def create_store(docs, name):
    if not docs:
        return None

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("mistral_api")
    )

    return Chroma.from_documents(docs, embeddings, collection_name=name)

# ════════════════════════════════════════════
# PDF
# ════════════════════════════════════════════

def extract_pdf_text(path):
    docs = PyPDFLoader(path).load()
    return split_docs(docs)

def extract_images(path):
    images = []
    doc = fitz.open(path)

    for page_num in range(len(doc)):
        for img in doc[page_num].get_images(full=True):
            base = doc.extract_image(img[0])
            images.append({
                "page": page_num,
                "image_bytes": base["image"]
            })
    return images

def process_images(images):
    key = os.getenv("vm_api")
    if not key:
        return []

    client = genai.Client(api_key=key)
    os.makedirs("images", exist_ok=True)

    docs = []

    for i, img in enumerate(images):
        try:
            path = f"images/img_{i}.png"

            with open(path, "wb") as f:
                f.write(img["image_bytes"])

            pil = PILImage.open(io.BytesIO(img["image_bytes"]))

            res = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[pil, "Describe this image"]
            )

            text = res.text if res else "No description"

        except Exception as e:
            text = f"Image failed: {e}"
            path = None

        docs.append(Document(page_content=text, metadata={"image_path": path}))

    return docs

# ════════════════════════════════════════════
# URL
# ════════════════════════════════════════════

def load_url(url):
    docs = WebBaseLoader(url).load()
    return split_docs(docs)

# ════════════════════════════════════════════
# GRAPH
# ════════════════════════════════════════════

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: list(x)+list(y)]
    images: list

def build_graph(tools):

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

# ════════════════════════════════════════════
# UI
# ════════════════════════════════════════════

st.title("🤖 Hybrid RAG Chatbot")

mode = st.radio("Choose Mode", ["PDF", "URL", "BOTH"])

pdf_file = st.file_uploader("Upload PDF", type="pdf")
url = st.text_input("Enter URL")

if st.button("Process"):

    tools = []

    # PDF
    if mode in ["PDF", "BOTH"] and pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            path = tmp.name

        text_docs = extract_pdf_text(path)
        img_docs = process_images(extract_images(path))

        text_db = create_store(text_docs, "pdf_text")
        img_db = create_store(img_docs, "pdf_img")

        if text_db:
            tools.append(create_retriever_tool(
                text_db.as_retriever(),
                name="pdf_text",
                description="PDF text retrieval"
            ))

        if img_db:
            tools.append(create_retriever_tool(
                img_db.as_retriever(),
                name="pdf_image",
                description="PDF image retrieval"
            ))

        st.session_state["img_db"] = img_db

    # URL
    if mode in ["URL", "BOTH"] and url:
        url_docs = load_url(url)
        url_db = create_store(url_docs, "url")

        tools.append(create_retriever_tool(
            url_db.as_retriever(),
            name="url",
            description="URL retrieval"
        ))

    # Tavily always fallback
    tools.append(tavily_tool)

    st.session_state.graph = build_graph(tools)
    st.success("✅ Ready")

# ════════════════════════════════════════════
# CHAT + IMAGE DISPLAY
# ════════════════════════════════════════════

if "graph" in st.session_state:
    q = st.text_input("Ask your question")

    if q:
        res = st.session_state.graph.invoke({
            "messages": [HumanMessage(content=q)],
            "images": []
        })

        answer = res["messages"][-1].content
        st.write("### 🤖 Answer")
        st.write(answer)

        # 🔥 IMAGE DISPLAY
        if "img_db" in st.session_state and st.session_state["img_db"]:
            docs = st.session_state["img_db"].as_retriever().invoke(q)

            imgs = [d.metadata.get("image_path") for d in docs if d.metadata.get("image_path")]

            if imgs:
                st.write("### 🖼️ Related Images")
                cols = st.columns(min(3, len(imgs)))

                for i, img in enumerate(imgs):
                    if os.path.exists(img):
                        cols[i % 3].image(img, use_container_width=True)
