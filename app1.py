import streamlit as st
import tempfile
import os
import io
import json

from dotenv import load_dotenv
load_dotenv()

# ─── LangChain core ───────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import create_retriever_tool
from langchain.chat_models import init_chat_model

# ─── LangChain community ──────────────────────────────────────────────────────
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings          # ← CHANGED
from langchain_chroma import Chroma

# ─── LangGraph ────────────────────────────────────────────────────────────────
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

# ─── Tavily ───────────────────────────────────────────────────────────────────
from langchain_tavily import TavilySearch

# ─── PDF / Image ──────────────────────────────────────────────────────────────
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import fitz
from PIL import Image as PILImage

# ─── Gemini (image summarisation) ────────────────────────────────────────────
from google import genai

# ─── Typing ───────────────────────────────────────────────────────────────────
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# ══════════════════════════════════════════════════════════════════════════════
# 🔧  SHARED CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# EMBED_MODEL constant removed — Mistral uses its own hosted model "mistral-embed"

llm = init_chat_model(
    model="groq:openai/gpt-oss-120b",
    api_key=os.getenv("llm_api")
)

tavily_tool = TavilySearch(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_images=True,
    name="tavily_search",
    description="if the user query is other than uploaded PDF or URL then only use this tool to Search the web for real-time information and any current information.always check with current dat and time while using this tool.",
)

# ══════════════════════════════════════════════════════════════════════════════
# 📄  PDF HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def partition_document(file_path: str):
    elements = partition_pdf(filename=file_path, strategy="fast")
    images = []
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        for img in doc[page_num].get_images(full=True):
            base_image = doc.extract_image(img[0])
            images.append({
                "page": page_num,
                "image_bytes": base_image["image"],
                "ext": base_image["ext"],
            })
    return elements, images


def flatten_elements(elements):
    flat = []
    for el in elements:
        if isinstance(el, list):
            flat.extend(flatten_elements(el))
        else:
            flat.append(el)
    return flat


def batch_chunking(elements):
    all_chunks = []
    for i in range(0, len(elements), 50):
        batch = elements[i : i + 50]
        chunks = chunk_by_title(
            batch,
            max_characters=1200,
            new_after_n_chars=800,
            combine_text_under_n_chars=200,
        )
        all_chunks.extend(chunks)

    texts = [
        chunk.text if hasattr(chunk, "text") else str(chunk) for chunk in all_chunks
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.create_documents(texts)


def chunks_to_documents(chunks):
    return [
        Document(page_content=c.page_content, metadata={"source": "pdf"})
        for c in chunks
    ]


def process_images_gemini(images):
    """Summarise extracted PDF images with Gemini Vision."""
    client = genai.Client(api_key=os.getenv("vm_api"))
    os.makedirs("images", exist_ok=True)
    img_docs = []
    for i, img in enumerate(images):
        image_path = os.path.abspath(f"images/img_{i}.png")
        with open(image_path, "wb") as f:
            f.write(img["image_bytes"])
        pil_img = PILImage.open(io.BytesIO(img["image_bytes"]))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                pil_img,
                "Describe this image in detail including diagrams, labels, text, and meaning",
            ],
        )
        img_docs.append(
            Document(
                page_content=response.text,
                metadata={"page": img["page"], "image_path": image_path},
            )
        )
    return img_docs


def create_chroma_store(documents, collection_name):
    embeddings = MistralAIEmbeddings(                        # ← CHANGED
        model="mistral-embed",
        api_key=os.getenv("mistral_api")
    )
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 🌐  URL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_url_retriever(url: str):
    """Load a URL, chunk it, embed into FAISS, return a retriever."""
    docs = WebBaseLoader(url).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = MistralAIEmbeddings(                        # ← CHANGED
        model="mistral-embed",
        api_key=os.getenv("mistral_api")
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever()


# ══════════════════════════════════════════════════════════════════════════════
# 🤖  AGENTIC RAG GRAPH
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: list(x) + list(y)]
    images: list
    tool_used: str


def build_graph(retriever_tool_obj, img_retriever_tool_obj=None):
    tools = [retriever_tool_obj, tavily_tool]
    if img_retriever_tool_obj is not None:
        tools.append(img_retriever_tool_obj)

    def agent(state: AgentState):
        print("---CALL AGENT---")
        model = llm.bind_tools(tools)
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
        print("---CHECK RELEVANCE---")

        class Grade(BaseModel):
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        grader = llm.with_structured_output(Grade)
        prompt = PromptTemplate(
            template=(
                "You are a grader assessing relevance of a retrieved document to a user question.\n"
                "Retrieved document:\n{context}\n\n"
                "User question: {question}\n"
                "Give a binary score 'yes' or 'no'."
            ),
            input_variables=["context", "question"],
        )
        chain = prompt | grader

        messages = state["messages"]
        question = messages[0].content
        docs = messages[-1].content

        result = chain.invoke({"question": question, "context": docs})
        if result.binary_score == "yes":
            print("---DOCS RELEVANT---")
            return "generate"
        print("---DOCS NOT RELEVANT---")
        return "rewrite"

    def generate(state: AgentState):
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content

        tool_name = None
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_name = msg.tool_calls[0]["name"]

        docs_raw = messages[-1].content
        context = docs_raw
        tavily_images = []

        try:
            docs_json = json.loads(docs_raw)
            context = docs_json.get("answer", docs_raw)
            tavily_images = docs_json.get("images", [])
        except (json.JSONDecodeError, TypeError):
            pass

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert assistant. Answer clearly and concisely using only the provided context. dont give lenghty answers, answers should short and precise"),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ])
        rag_chain = answer_prompt | llm | StrOutputParser()
        answer = rag_chain.invoke({"context": context, "question": question})

        retrieved_images = []
        if tool_name == retriever_tool_obj.name and img_retriever_tool_obj is not None:
            img_docs = img_retriever_tool_obj.func(question)
            for d in img_docs:
                path = d.metadata.get("image_path")
                if path:
                    retrieved_images.append(path)
        else:
            retrieved_images = tavily_images

        return {
            "messages": [AIMessage(content=answer)],
            "images": retrieved_images,
            "tool_used": tool_name or "unknown",
        }

    def rewrite(state: AgentState):
        print("---TRANSFORM QUERY---")
        question = state["messages"][0].content
        msg = [
            HumanMessage(
                content=(
                    "Look at this question and reason about its semantic intent.\n"
                    f"Original question:\n{question}\n\n"
                    "Formulate an improved question:"
                )
            )
        ]
        response = llm.invoke(msg)
        return {"messages": [response]}

    wf = StateGraph(AgentState)
    wf.add_node("agent", agent)
    wf.add_node("retrieve", ToolNode(tools))
    wf.add_node("rewrite", rewrite)
    wf.add_node("generate", generate)

    wf.add_edge(START, "agent")
    wf.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
    wf.add_conditional_edges("retrieve", grade_documents)
    wf.add_edge("generate", END)
    wf.add_edge("rewrite", "agent")

    return wf.compile()


# ══════════════════════════════════════════════════════════════════════════════
# 🔍  INTENT DETECTION  (image requests)
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_KEYWORDS = {
    "diagram", "architecture", "flowchart", "visual", "draw",
    "illustration", "image", "figure", "graph", "plot", "show",
}

def wants_images(query: str) -> bool:
    return any(kw in query.lower() for kw in IMAGE_KEYWORDS)


# ══════════════════════════════════════════════════════════════════════════════
# 💬  QUERY RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_query(query: str, graph, image_retriever=None):
    result = graph.invoke({"messages": [HumanMessage(content=query)], "images": []})

    answer = result["messages"][-1].content
    images = result.get("images", [])
    tool_used = result.get("tool_used", "unknown")

    if image_retriever and wants_images(query):
        img_docs = image_retriever.invoke(query)
        for d in img_docs:
            path = d.metadata.get("image_path")
            if path and path not in images:
                images.append(path)

    return answer, images, tool_used

# ══════════════════════════════════════════════════════════════════════════════
# 🚀  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Multi Modal Agentic RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Multi Modal Agentic RAG Chatbot")
st.caption("Ask questions from a **PDF document** or a **web URL** — powered by LangGraph + Tavily")

source_mode = st.radio("Choose input source:", ["📄 PDF Upload", "🌐 Web URL"], horizontal=True)

for key in ("graph", "image_retriever", "ready"):
    if key not in st.session_state:
        st.session_state[key] = None
st.session_state.setdefault("ready", False)

# ══════════════════════════════════════════════════════════════════════════════
# 📄  PDF MODE
# ══════════════════════════════════════════════════════════════════════════════
if source_mode == "📄 PDF Upload":
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file and st.button("⚙️ Process PDF"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        with st.spinner("Partitioning and chunking PDF…"):
            elements, raw_images = partition_document(file_path)
            elements = flatten_elements(elements)
            chunks = batch_chunking(elements)
            text_docs = chunks_to_documents(chunks)

        with st.spinner("Building text vector store…"):
            text_db = create_chroma_store(text_docs, "pdf_text")
            text_retriever = text_db.as_retriever(
                search_type="mmr", search_kwargs={"k": 5}
            )

        with st.spinner("Summarising images with Gemini Vision…"):
            img_docs = process_images_gemini(raw_images)

        with st.spinner("Building image vector store…"):
            image_db = create_chroma_store(img_docs, "pdf_images")
            image_retriever = image_db.as_retriever(
                search_type="mmr", search_kwargs={"k": 1}
            )

        pdf_retriever_tool = create_retriever_tool(
            text_retriever,
            name="pdf_retriever",
            description="Search and retrieve information from the uploaded PDF document",
        )
        image_retriever_tool = create_retriever_tool(
            image_retriever,
            name="image_retriever",
            description="if the user query demands retrieval of images, diagrams, flowcharts, visual explanations from uploaded PDF then only use this tool to retrieve relevant images docs from then PDF, otherwise do not use this tool. "
        )

        with st.spinner("Compiling agentic graph…"):
            st.session_state["graph"] = build_graph(pdf_retriever_tool, image_retriever_tool)
            st.session_state["image_retriever"] = image_retriever
            st.session_state["ready"] = True

        st.success("✅ PDF processed and ready!")

# ══════════════════════════════════════════════════════════════════════════════
# 🌐  URL MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    url_input = st.text_input("Enter a webpage URL", placeholder="https://en.wikipedia.org/wiki/...")

    if url_input and st.button("⚙️ Process URL"):
        with st.spinner("Loading and indexing URL…"):
            url_retriever = build_url_retriever(url_input)

        url_retriever_tool = create_retriever_tool(
            url_retriever,
            name="url_retriever",
            description=f"Search and retrieve information from: {url_input}",
        )

        with st.spinner("Compiling agentic graph…"):
            st.session_state["graph"] = build_graph(url_retriever_tool)
            st.session_state["image_retriever"] = None
            st.session_state["ready"] = True

        st.success("✅ URL indexed and ready!")

# ══════════════════════════════════════════════════════════════════════════════
# 💬  CHAT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("ready"):
    st.divider()
    query = st.text_input("💬 Ask your question")

    if query:
        with st.spinner("Thinking…"):
            answer, images, tool_used = run_query(
                query,
                st.session_state["graph"],
                st.session_state["image_retriever"],
            )

        st.write("### 🤖 Answer")
        st.write(answer)

        st.write(f"### 🛠️ Tool Used: {tool_used}")

        if images:
            st.write("### 🖼️ Related Images")
            cols = st.columns(min(len(images), 3))
            for i, img in enumerate(images):
                with cols[i % 3]:
                    if img.startswith("http"):
                        st.image(img, use_container_width=True)
                    elif os.path.exists(img):
                        st.image(img, use_container_width=True)
                    else:
                        st.caption(f"Image not found: {img}")