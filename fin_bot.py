import streamlit as st
from langchain_openai import ChatOpenAI 
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import ssl
import os

# Set your OpenAI API key here

# Define helper functions
def get_sitemap(url):
    req = Request(
        url=url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; YourAppName/1.0; +http://yourdomain.com)"}
    )
    response = urlopen(req)
    xml = BeautifulSoup(
        response,
        "lxml-xml",
        from_encoding=response.info().get_param("charset")
    )
    return xml

def get_urls(xml):
    urls = []
    for url in xml.find_all("url"):
        if xml.find("loc"):
            loc = url.findNext("loc").text
            urls.append(loc)
    return urls

def scrape_site(url="https://zerodha.com/varsity/chapter-sitemap2.xml"):
    ssl._create_default_https_context = ssl._create_stdlib_context
    xml = get_sitemap(url)
    urls = get_urls(xml)

    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    return docs

def vector_retriever(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore.as_retriever()

def run_chain(question):
    docs = scrape_site()
    retriever = vector_retriever(docs)
    
    system_prompt = (
        "You are a financial assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise. If the question is not clear ask follow-up questions.\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain.invoke({"input": question})

# Streamlit UI
st.title("Financial Assistant")
st.write("Enter a question below to get answers from the financial assistant:")

# Text input field for the question
question = st.text_input("Your Question:")

# If question is provided, process it and display the answer
if question:
    with st.spinner("Processing..."):
        response = run_chain(question)
        st.write("### Answer:")
        st.write(response["answer"])


