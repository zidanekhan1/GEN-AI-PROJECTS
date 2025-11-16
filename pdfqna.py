import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
llm = ChatGroq(model="llama-3.3-70b-versatile")
prompt = ChatPromptTemplate.from_messages([
    (
    "system",
    """
    Answer the questions based on the provided context only
    provide the most accurate answers
    <context>
    {context}
    <context>
    """
    ),
    (
        "human",
        "Question:{input}"
    )

])

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = embeddings
        st.session_state.loader=PyPDFDirectoryLoader("pdfiles")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=100)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

user_prompt=st.text_input("Enter your query from the pedophiles")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("vector database ready")

import time
if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({"input":user_prompt})
    print(f"response time : {time.process_time()-start}")

    st.write(response['answer'])

    with st.expander("Document similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("------------------------------")
