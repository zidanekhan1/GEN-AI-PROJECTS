from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

##langsmith tracker
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A BOT WITH OLLAMA"

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","provide good, and consise answers in two to three sentences"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,engine,temperature,max_tokens):

    llm=ChatOllama(model=engine)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer

engine=st.sidebar.selectbox("select ollama model:",["granite3.1-moe:1b","deepseek-r1:1.5b","gemma3:1b"])
temperature=st.sidebar.slider("set temperature",min_value=0.1,max_value=1.0,value=0.6)
max_tokens=st.sidebar.slider("select max tokens",min_value=50,max_value=300,value=120)

st.write("ask anything you want comrade")
user_input=st.text_input("you:")

if user_input:
    response=generate_response(user_input,engine,temperature,max_tokens)
    st.write(response)

else:
    st.write("whats the matter?")