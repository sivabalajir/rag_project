import streamlit as st
from dotenv import load_dotenv
import os
import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ============================================================
# LOAD API KEY + LANGSMITH SETUP
# ============================================================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# ============================================================
# GUARDRAILS
# ============================================================
def check_pii(question):
    pii_patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        r'\b\d{10}\b',
        r'\bsalary of\b',
        r'\bpayroll of\b',
        r'\bpersonal details of\b',
        r'\baddress of\b',
    ]
    question_lower = question.lower()
    for pattern in pii_patterns:
        if re.search(pattern, question_lower):
            return True
    return False

def check_out_of_scope(question):
    out_of_scope_keywords = [
        "cricket", "ipl", "football", "movie", "film",
        "recipe", "cook", "weather", "politics", "election",
        "stock market", "bitcoin", "crypto", "game", "sport"
    ]
    question_lower = question.lower()
    for keyword in out_of_scope_keywords:
        if keyword in question_lower:
            return True
    return False

def run_guardrails(question):
    if check_pii(question):
        return False, "⚠️ I cannot answer questions containing personal/private information."
    if check_out_of_scope(question):
        return False, "⚠️ I can only answer company related questions. This question is out of scope."
    return True, None

# ============================================================
# RBAC
# ============================================================
ROLE_ACCESS = {
    "hr":          ["hr", "general"],
    "finance":     ["finance", "general"],
    "engineering": ["engineering", "general"],
    "marketing":   ["marketing", "general"],
    "ceo":         ["hr", "finance", "engineering", "marketing", "general"]
}

# ============================================================
# LOAD VECTORSTORE - only load once using Streamlit cache
# ============================================================
# @st.cache_resource means this function runs only ONCE
# even if user refreshes the page — saves time and memory!
@st.cache_resource
def load_vectorstore():
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    for department in ["hr", "finance", "engineering", "marketing", "general"]:
        folder_path = f"data/{department}"
        if not os.path.exists(folder_path):
            continue
        loader = DirectoryLoader(
            folder_path,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        try:
            docs = loader.load()
        except Exception:
            continue
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["department"] = department
        all_chunks.extend(chunks)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory="db"
    )
    return vectorstore

# ============================================================
# STREAMLIT UI
# ============================================================

# Page title and icon
st.set_page_config(page_title="Snow Bot", page_icon="❄️")
st.title("❄️ Snow Bot")
st.caption("Ask questions about company data based on your role!")

# Role selector in sidebar
# sidebar = left panel in Streamlit
with st.sidebar:
    st.header("👤 Login")
    # Dropdown to select role
    role = st.selectbox(
        "Select your role:",
        ["hr", "finance", "engineering", "marketing", "ceo"]
    )
    st.info(f"You are logged in as: **{role}**")
    st.write("**Your access:**")
    for dept in ROLE_ACCESS[role]:
        st.write(f"  ✅ {dept}")

# Initialize chat history
# st.session_state stores data between reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box at bottom
if prompt := st.chat_input("Ask a question..."):

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run guardrails
    is_safe, block_message = run_guardrails(prompt)

    if not is_safe:
        # Show blocked message
        with st.chat_message("assistant"):
            st.warning(block_message)
        st.session_state.messages.append({"role": "assistant", "content": block_message})

    else:
        # Get answer from RAG
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                vectorstore = load_vectorstore()

                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    api_key=groq_api_key
                )

                allowed_departments = ROLE_ACCESS.get(role, [])

                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": 3,
                        "filter": {"department": {"$in": allowed_departments}}
                    }
                )

                template = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

Context: {context}

Question: {question}
""")

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | template
                    | llm
                    | StrOutputParser()
                )

                answer = chain.invoke(prompt)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})