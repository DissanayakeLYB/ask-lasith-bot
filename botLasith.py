import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OpenAI_API")


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_models import ChatOpenAI

 
st.header("PDFTalk")

file = st.file_uploader("Upload a PDF File and start asking questions : ", type="pdf")
with st.sidebar:
    st.title("Chat History")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        separators = "\n",
        chunk_size = 500,
        chunk_overlap = 150,
    )

    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

    vector_store = FAISS.from_texts(chunks, embeddings)

    user_question = st.chat_input("Ask the question...")

    

    if user_question:
        match = vector_store.similarity_search(user_question) 

        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0, 
            max_tokens = 1000
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        chain = create_history_aware_retriever(llm = llm, retriever = retriever, prompt = user_question)

        response = chain.run(input_documents= match, question = user_question)

        # # display response of the assistant
        with st.chat_message("user"):
            st.write(user_question)

        # add assistant's response to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})

        response = llm.invoke(user_question)

        # display response of the assistant
        with st.chat_message("assistant"):
            st.write(response.content)

        # add assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.content})