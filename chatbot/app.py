LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="<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>"
LANGSMITH_PROJECT="Chatbot_Llama"


from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
import streamlit as st 
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"]="True"
api_key = os.getenv("LANGCHAIN_API_KEY")
if api_key is None:
    raise ValueError("LANGCHAIN_API_KEY is missing. Check your .env file.")

os.environ["LANGCHAIN_API_KEY"] = api_key

## Creating Chatbot

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user", "Question:{question}")
    ]    
)

## streamlit framework

st.title('Langchain Demo With LLAMA2 API')
input_text=st.text_input("Search the topic you want")

# ollama LLAma2 LLm 
llm=Ollama(model="llama3.2")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

# Test the connection
try:
    response = llm.invoke("Test")
    print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")

if input_text:
    st.write(chain.invoke({"question":input_text}))
