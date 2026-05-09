import streamlit as st
import tempfile
import os
import io
import json

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────
# LangChain Core
# ─────────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    BaseMessage,
    ToolMessage
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import create_retriever_tool

from langchain.chat_models import init_chat_model

# ─────────────────────────────────────────────────────────────
# Community
# ─────────────────────────────────────────────────────────────
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader
)

from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

# ─────────────────────────────────────────────────────────────
# LangGraph
# ─────────────────────────────────────────────────────────────
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

# ─────────────────────────────────────────────────────────────
# Tavily
# ─────────────────────────────────────────────────────────────
from langchain_tavily import TavilySearch

# ─────────────────────────────────────────────────────────────
# PDF / Images
# ─────────────────────────────────────────────────────────────
import fitz
from PIL import Image as PILImage

# ─────────────────────────────────────────────────────────────
# Gemini
# ─────────────────────────────────────────────────────────────
from google import genai

# ─────────────────────────────────────────────────────────────
# Typing
# ─────────────────────────────────────────────────────────────
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# ═════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Multi Modal Agentic RAG",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi Modal Agentic RAG Chatbot")
st.caption("PDF + URL + Image Retrieval + Agentic RAG")

# ═════════════════════════════════════════════════════════════
# LLM
# ═════════════════════════════════════════════════════════════

llm = init_chat_model(
    model="groq:openai/gpt-oss-120b",
    api_key=os.getenv("llm_api")
)

# ═════════════════════════════════════════════════════════════
# TAVILY
# ═════════════════════════════════════════════════════════════

tavily_tool = TavilySearch(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_images=True,
    name="tavily_search",
    description="Search realtime web information"
)

# ═════════════════════════════════════════════════════════════
# EMBEDDINGS
# ═════════════════════════════════════════════════════════════

def get_embeddings():

    return MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("mistral_api")
    )

# ═════════════════════════════════════════════════════════════
# PDF TEXT EXTRACTION
# ═════════════════════════════════════════════════════════════

def extract_pdf_text(file_path):

    try:

        loader = PyPDFLoader(file_path)

        docs = loader.load()

        if not docs:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = splitter.split_documents(docs)

        clean_docs = []

        for d in splits:

            text = d.page_content.strip()

            if len(text) > 20:

                clean_docs.append(
                    Document(
                        page_content=text,
                        metadata=d.metadata
                    )
                )

        return clean_docs

    except Exception as e:

        st.error(f"PDF extraction failed: {e}")

        return []

# ═════════════════════════════════════════════════════════════
# IMAGE EXTRACTION
# ═════════════════════════════════════════════════════════════

def extract_images(file_path):

    images = []

    try:

        doc = fitz.open(file_path)

        for page_num in range(len(doc)):

            for img in doc[page_num].get_images(full=True):

                try:

                    base_image = doc.extract_image(img[0])

                    images.append({
                        "page": page_num,
                        "image_bytes": base_image["image"],
                        "ext": base_image["ext"]
                    })

                except:
                    pass

    except Exception as e:

        print("Image extraction error:", e)

    return images

# ═════════════════════════════════════════════════════════════
# GEMINI IMAGE PROCESSING
# ═════════════════════════════════════════════════════════════

def process_images_gemini(images):

    if not images:
        return []

    client = genai.Client(
        api_key=os.getenv("vm_api")
    )

    os.makedirs("images", exist_ok=True)

    docs = []

    for i, img in enumerate(images):

        try:

            image_path = os.path.abspath(
                f"images/img_{i}.png"
            )

            with open(image_path, "wb") as f:
                f.write(img["image_bytes"])

            pil_img = PILImage.open(
                io.BytesIO(img["image_bytes"])
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    pil_img,
                    "Describe this image in detail including diagrams, labels, equations, architecture and meaning"
                ]
            )

            text = ""

            try:
                text = response.text
            except:
                text = "No description"

            if text:

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "page": img["page"],
                            "image_path": image_path
                        }
                    )
                )

        except Exception as e:

            print("Gemini image failed:", e)

            continue

    return docs

# ═════════════════════════════════════════════════════════════
# VECTOR STORE
# ═════════════════════════════════════════════════════════════

def create_chroma_store(documents, collection_name):

    valid_docs = []

    for d in documents:

        if d.page_content and len(d.page_content.strip()) > 5:

            valid_docs.append(d)

    if len(valid_docs) == 0:

        raise ValueError("No valid documents found")

    embeddings = get_embeddings()

    return Chroma.from_documents(
        documents=valid_docs,
        embedding=embeddings,
        collection_name=collection_name
    )

# ═════════════════════════════════════════════════════════════
# URL RETRIEVER
# ═════════════════════════════════════════════════════════════

def build_url_retriever(url):

    loader = WebBaseLoader(url)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = splitter.split_documents(docs)

    clean_docs = []

    for d in splits:

        text = d.page_content.strip()

        if len(text) > 20:

            clean_docs.append(d)

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        clean_docs,
        embeddings
    )

    return vectorstore.as_retriever()

# ═════════════════════════════════════════════════════════════
# AGENT STATE
# ═════════════════════════════════════════════════════════════

class AgentState(TypedDict):

    messages: Annotated[
        Sequence[BaseMessage],
        lambda x, y: list(x) + list(y)
    ]

    images: list

    tool_used: str

# ═════════════════════════════════════════════════════════════
# GRAPH
# ═════════════════════════════════════════════════════════════

def build_graph(pdf_tool, image_tool=None):

    tools = [pdf_tool, tavily_tool]

    if image_tool:
        tools.append(image_tool)

    # ─────────────────────────────────────────

    def agent(state):

        model = llm.bind_tools(tools)

        response = model.invoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }

    # ─────────────────────────────────────────

    def grade_documents(state):

        try:

            class Grade(BaseModel):

                binary_score: str = Field(
                    description="yes or no"
                )

            grader = llm.with_structured_output(Grade)

            question = state["messages"][0].content

            docs = ""

            for msg in reversed(state["messages"]):

                if isinstance(msg, ToolMessage):

                    docs = str(msg.content)

                    break

            docs = docs[:4000]

            prompt = PromptTemplate(
                template="""
You are grading relevance.

Document:
{context}

Question:
{question}

Reply only:
yes
or
no
""",
                input_variables=[
                    "context",
                    "question"
                ]
            )

            chain = prompt | grader

            result = chain.invoke({
                "question": question,
                "context": docs
            })

            if result.binary_score.lower() == "yes":

                return "generate"

            return "rewrite"

        except Exception as e:

            print("Grading failed:", e)

            return "generate"

    # ─────────────────────────────────────────

    def rewrite(state):

        question = state["messages"][0].content

        msg = HumanMessage(
            content=f"""
Rewrite this question more clearly:

{question}
"""
        )

        response = llm.invoke([msg])

        return {
            "messages": [response]
        }

    # ─────────────────────────────────────────

    def generate(state):

        question = state["messages"][0].content

        context = ""

        tool_name = "unknown"

        images = []

        # tool name

        for msg in state["messages"]:

            if hasattr(msg, "tool_calls") and msg.tool_calls:

                try:

                    tool_name = msg.tool_calls[0]["name"]

                except:
                    pass

        # get tool content safely

        for msg in reversed(state["messages"]):

            if isinstance(msg, ToolMessage):

                context = str(msg.content)

                break

        context = context[:12000]

        # tavily json parsing

        try:

            docs_json = json.loads(context)

            if isinstance(docs_json, dict):

                context = docs_json.get(
                    "answer",
                    context
                )

                images = docs_json.get(
                    "images",
                    []
                )

        except:
            pass

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are an expert AI assistant.

Answer clearly using ONLY the provided context.

Keep answers concise and accurate.
"""
            ),
            (
                "human",
                """
Context:
{context}

Question:
{question}
"""
            )
        ])

        chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "messages": [
                AIMessage(content=answer)
            ],
            "images": images,
            "tool_used": tool_name
        }

    # ─────────────────────────────────────────

    wf = StateGraph(AgentState)

    wf.add_node(
        "agent",
        agent
    )

    wf.add_node(
        "retrieve",
        ToolNode(tools)
    )

    wf.add_node(
        "rewrite",
        rewrite
    )

    wf.add_node(
        "generate",
        generate
    )

    wf.add_edge(
        START,
        "agent"
    )

    wf.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END
        }
    )

    wf.add_conditional_edges(
        "retrieve",
        grade_documents
    )

    wf.add_edge(
        "generate",
        END
    )

    wf.add_edge(
        "rewrite",
        "agent"
    )

    return wf.compile()

# ═════════════════════════════════════════════════════════════
# IMAGE QUERY DETECTION
# ═════════════════════════════════════════════════════════════

IMAGE_KEYWORDS = {
    "diagram",
    "architecture",
    "flowchart",
    "image",
    "figure",
    "visual",
    "graph",
    "plot",
    "show images",
    "with images"
}

def wants_images(query):

    q = query.lower()

    return any(
        word in q
        for word in IMAGE_KEYWORDS
    )

# ═════════════════════════════════════════════════════════════
# QUERY RUNNER
# ═════════════════════════════════════════════════════════════

def run_query(query, graph, image_retriever=None):

    result = graph.invoke({
        "messages": [
            HumanMessage(content=query)
        ],
        "images": []
    })

    answer = result["messages"][-1].content

    images = result.get(
        "images",
        []
    )

    tool_used = result.get(
        "tool_used",
        "unknown"
    )

    # retrieve pdf images

    if image_retriever and wants_images(query):

        try:

            img_docs = image_retriever.invoke(query)

            for d in img_docs:

                path = d.metadata.get(
                    "image_path"
                )

                if path and path not in images:

                    images.append(path)

        except Exception as e:

            print("Image retriever failed:", e)

    return answer, images, tool_used

# ═════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════

for key in [
    "graph",
    "image_retriever",
    "ready"
]:
    if key not in st.session_state:
        st.session_state[key] = None

st.session_state.setdefault(
    "ready",
    False
)

# ═════════════════════════════════════════════════════════════
# SOURCE MODE
# ═════════════════════════════════════════════════════════════

source_mode = st.radio(
    "Choose Input Source",
    [
        "📄 PDF Upload",
        "🌐 Web URL"
    ],
    horizontal=True
)

# ═════════════════════════════════════════════════════════════
# PDF MODE
# ═════════════════════════════════════════════════════════════

if source_mode == "📄 PDF Upload":

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf"
    )

    if uploaded_file and st.button("Process PDF"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.read())

            file_path = tmp.name

        # ─────────────────────────────────────

        with st.spinner("Reading PDF..."):

            text_docs = extract_pdf_text(
                file_path
            )

        st.write(
            "Chunks:",
            len(text_docs)
        )

        if len(text_docs) == 0:

            st.error(
                "❌ No text extracted from PDF"
            )

            st.stop()

        # ─────────────────────────────────────

        with st.spinner("Extracting images..."):

            raw_images = extract_images(
                file_path
            )

        # ─────────────────────────────────────

        with st.spinner("Processing images with Gemini..."):

            img_docs = process_images_gemini(
                raw_images
            )

        # ─────────────────────────────────────

        with st.spinner("Building vector DB..."):

            text_db = create_chroma_store(
                text_docs,
                "pdf_text"
            )

            text_retriever = text_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5}
            )

            image_retriever = None

            if img_docs:

                img_db = create_chroma_store(
                    img_docs,
                    "pdf_images"
                )

                image_retriever = img_db.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 2}
                )

        # ─────────────────────────────────────

        pdf_tool = create_retriever_tool(
            text_retriever,
            name="pdf_retriever",
            description="Retrieve information from uploaded PDF"
        )

        image_tool = None

        if image_retriever:

            image_tool = create_retriever_tool(
                image_retriever,
                name="image_retriever",
                description="Retrieve diagrams and images from PDF"
            )

        # ─────────────────────────────────────

        with st.spinner("Building graph..."):

            st.session_state["graph"] = build_graph(
                pdf_tool,
                image_tool
            )

            st.session_state["image_retriever"] = image_retriever

            st.session_state["ready"] = True

        st.success("✅ PDF Ready!")

# ═════════════════════════════════════════════════════════════
# URL MODE
# ═════════════════════════════════════════════════════════════

else:

    url = st.text_input(
        "Enter URL"
    )

    if url and st.button("Process URL"):

        with st.spinner("Loading URL..."):

            retriever = build_url_retriever(
                url
            )

        url_tool = create_retriever_tool(
            retriever,
            name="url_retriever",
            description=f"Retrieve information from {url}"
        )

        with st.spinner("Building graph..."):

            st.session_state["graph"] = build_graph(
                url_tool
            )

            st.session_state["image_retriever"] = None

            st.session_state["ready"] = True

        st.success("✅ URL Ready!")

# ═════════════════════════════════════════════════════════════
# CHAT
# ═════════════════════════════════════════════════════════════

if st.session_state.get("ready"):

    st.divider()

    query = st.text_input(
        "Ask your question"
    )

    if query:

        with st.spinner("Thinking..."):

            answer, images, tool_used = run_query(
                query,
                st.session_state["graph"],
                st.session_state["image_retriever"]
            )

        st.write("## 🤖 Answer")

        st.write(answer)

        st.write(f"### 🛠️ Tool Used: {tool_used}")

        # images

        if images:

            st.write("## 🖼️ Related Images")

            cols = st.columns(
                min(len(images), 3)
            )

            for i, img in enumerate(images):

                with cols[i % 3]:

                    try:

                        if isinstance(img, str):

                            if img.startswith("http"):

                                st.image(
                                    img,
                                    use_container_width=True
                                )

                            elif os.path.exists(img):

                                st.image(
                                    img,
                                    use_container_width=True
                                )

                    except:
                        pass
