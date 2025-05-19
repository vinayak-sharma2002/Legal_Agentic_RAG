import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Cassandra
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_core.documents.base import Document
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import cassio
from langchain_community.tools.tavily_search import TavilySearchResults



# groq_api_key = os.getenv("GROQ_API_KEY")
# AstraDB_token = os.getenv("token")
# AstraDB_database_id = os.getenv("database_id")
# AstraDB_keyspace = os.getenv("keyspace")


# ---- Astra DB Init ----
cassio.init(
    token=os.getenv("token"),
    database_id=os.getenv("database_id"),
    keyspace=os.getenv("keyspace")
)

# ---- Setup ----
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"),model_name="llama3-70b-8192", temperature=0)
search_tool = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API"))

# ---- Document Loader ----
def load_and_split(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(docs)

# ---- Streamlit UI ----
st.set_page_config(page_title="Agentic RAG App", layout="wide")
st.title("Agentic RAG: Legal Docs + Web Search")

uploaded_files = st.file_uploader("Upload your legal documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    with st.spinner("Processing and indexing documents..."):
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name
            docs = load_and_split(tmp_path)
            all_docs.extend(docs)
            os.remove(tmp_path)

        vectorstore = Cassandra.from_documents(
            documents=all_docs,
            embedding=embeddings,
            table_name="uploaded_legal_docs"
        )

        st.success("Documents embedded into Astra DB!")

        # Define document search tool
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        

        # ---- Prompt Template for Summarization ----
        summary_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
        You are a legal assistant. Given the following case excerpts, answer the user's question.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """.strip()
        )

        # ---- LLM + Output Parser Chain using LCEL ----
        summary_chain = summary_prompt | llm | StrOutputParser()

        # ---- Tool Function using the LCEL chain ----
        def document_tool_func(query):
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            return summary_chain.invoke({"context": context, "question": query})

        # ---- Register the tool ----
        doc_tool = Tool(
            name="Document Search",
            func=document_tool_func,
            description="Use this tool to answer questions based on the uploaded legal documents. Use this tool whenever the user asks a question that is related to the uploaded documents. The tool will search the documents and return the answer.",
        )


        web_tool = Tool(
            name="Web Search",
            func=search_tool.run,
            description="Search the internet using Tavily Search, use this tool whenever the user asks a question that is not related to the uploaded documents, or if the user implies that the answer is not in the document. The tool will search the web and return the answer.",
        )

        agent = initialize_agent(
            tools=[doc_tool, web_tool],
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=50,
            max_execution_time=120,            
        )

        user_query = st.text_input("Ask your legal question")

        if user_query:
            with st.spinner("Thinking..."):
                response = agent.run(user_query)
                st.write("### Answer")
                st.write(response)
