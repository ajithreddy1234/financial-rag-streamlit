# financial_rag.py

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from datetime import datetime
import os

def load_and_chunk_docs(file_paths):
    documents = []
    for path in file_paths:
        loader = PyMuPDFLoader(path)
        documents += loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(documents)

def create_vector_db(chunks, persist_directory='./vector_db'):
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb

def load_or_create_vectordb(doc_paths, persist_directory='./vector_db'):
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    else:
        chunks = load_and_chunk_docs(doc_paths)
        return create_vector_db(chunks, persist_directory)

def rag_query(query, vectordb):
    llm = Ollama(model="Mistral")
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
    return qa_chain.run(query)

def web_search(query):
    search = DuckDuckGoSearchRun()
    return search.run(query)

def generate_report(rag_info, web_info, query):
    llm = Ollama(model="Mistral")
    current_datetime = datetime.now()
    template = PromptTemplate.from_template(
        """
        You are an elite research analyst in the financial services domain.
        Your expertise encompasses:

        - Deep investigative financial research and analysis
        - Fact-checking and source verification
        - Data-driven reporting and visualization
        - Expert interview synthesis
        - Trend analysis and future predictions
        - Complex topic simplification
        - Ethical practices
        - Balanced perspective presentation
        - Global context integration

        Research Phase:
        - Utilize internal and external data provided
        - Prioritize recent publications and expert opinions

        Analysis Phase:
        - Cross-reference facts
        - Identify emerging trends

        Writing Phase:
        - Craft an attention-grabbing headline
        - Structure in Financial Report style with executive summary, key findings, impact analysis, and future outlook
        - Include relevant quotes and statistics clearly

        Quality Control:
        - Verify facts and ensure narrative readability

        Internal Data:
        {rag_info}

        External Market Data:
        {web_info}

        # Financial Report on {query}

        ## Executive Summary
        
        ## Background & Context
        
        ## Key Findings
        
        ## Impact Analysis
        
        ## Future Outlook
        
        ## Expert Insights
        
        ## Sources & Methodology
        
        ---
        Research conducted by Financial Agent
        Published: {current_datetime:%Y-%m-%d}
        Last Updated: {current_datetime:%H:%M:%S}
        """
    )
    prompt = template.format(
        rag_info=rag_info, web_info=web_info, query=query, current_datetime=current_datetime
    )
    return llm.invoke(prompt)
