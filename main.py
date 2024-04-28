import os
from dotenv import load_dotenv
from groq import Groq

import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

client = Groq(
    api_key=os.getenv("groq_API"),
)

memory=ConversationBufferWindowMemory(k=10)

st.title("MSE Expert - Lasith")
input_meesage = st.chat_input("Ask from expert...")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history=[]

else:
    for message in st.session_state.chat_history:
        memory.save_context({'input':message['human']},{'output':message['AI']})

with st.sidebar:
    st.button(
         label = "New Chat",
         type="primary"
         )

    model = st.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )

    st.title("Chat history")

groq_chat = ChatGroq(
    groq_api_key=os.getenv("groq_API"), 
    model_name=model
)

conversation = ConversationChain(
    llm=groq_chat,
    memory=memory
)

if input_meesage:
    response = conversation(input_meesage)
    message = {'human':input_meesage,'AI':response['response']}
    st.session_state.chat_history.append(message)
    st.write(f"Human : {input_meesage}")
    st.write("AI : ", response['response'])

