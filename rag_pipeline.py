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
# LOAD API KEY
# ============================================================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ============================================================
# LANGSMITH MONITORING SETUP
# ============================================================
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# ============================================================
# GUARDRAIL 1 - PII DETECTION
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

# ============================================================
# GUARDRAIL 2 - OUT OF SCOPE DETECTION
# ============================================================
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

# ============================================================
# GUARDRAIL CHECK FUNCTION
# ============================================================
def run_guardrails(question):
    if check_pii(question):
        return False, "⚠️ I cannot answer questions containing personal/private information."
    if check_out_of_scope(question):
        return False, "⚠️ I can only answer company related questions. This question is out of scope."
    return True, None

# ============================================================
# RBAC - ROLE ACCESS MAP
# ============================================================
ROLE_ACCESS = {
    "hr":          ["hr", "general"],
    "finance":     ["finance", "general"],
    "engineering": ["engineering", "general"],
    "marketing":   ["marketing", "general"],
    "ceo":         ["hr", "finance", "engineering", "marketing", "general"]
}

# ============================================================
# STEP 1 - LOAD DOCUMENTS WITH METADATA
# ============================================================
print("Loading documents...")
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
    print(f"  {department}: {len(chunks)} chunks loaded")

print(f"Total chunks: {len(all_chunks)}")

# ============================================================
# STEP 2 - STORE IN CHROMADB WITH METADATA
# ============================================================
print("\nStoring in ChromaDB...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory="db"
)
print("Successfully stored in ChromaDB!")

# ============================================================
# STEP 3 - LLM SETUP
# ============================================================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key
)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

Context: {context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ============================================================
# STEP 4 - ASK QUESTION WITH RBAC + GUARDRAILS
# ============================================================
def ask_question(question, role):
    print(f"\nRole: {role}")
    print(f"Question: {question}")

    # 🛡️ Run guardrails FIRST
    is_safe, block_message = run_guardrails(question)
    if not is_safe:
        print(f"Blocked: {block_message}")
        print("-" * 50)
        return

    # 🔐 RBAC - get allowed departments
    allowed_departments = ROLE_ACCESS.get(role, [])
    print(f"Allowed departments: {allowed_departments}")

    # Search only in allowed departments
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
            "filter": {"department": {"$in": allowed_departments}}
        }
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    print(f"Answer: {answer}")
    print("-" * 50)

# ============================================================
# TEST ALL FEATURES
# ============================================================

# Valid question ✅
ask_question("What is the leave policy?", role="hr")

# PII question — blocked 🚫
ask_question("What is the salary of Rahul?", role="hr")

# Out of scope — blocked 🚫
ask_question("Who will win IPL this year?", role="hr")

# Finance valid question ✅
ask_question("What is the company revenue?", role="finance")