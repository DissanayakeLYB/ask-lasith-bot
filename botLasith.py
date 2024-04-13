#OpenAI key
OPENAI_API_KEY = #add the key here

#streamlit is for UI
import streamlit as st

#to work with PDF,
from PyPDF2 import PdfReader

#to make chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

#to generate embeddings
from langchain.embeddings.openai import OpenAIEmbeddings

#to call the database 
from langchain_community.vectorstores.faiss import FAISS

#chaintype load
from langchain.chains.question_answering import load_qa_chain

#import llm 
from langchain_community.chat_models import ChatOpenAI

 
st.header("Ask Lasith's PDF")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF File and start asking questions : ", type="pdf")

#extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        # st.write(text)


    #break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators = "\n",
        chunk_size = 500, #these must be updated until you get the best results
        chunk_overlap = 150, #this allows next chunk to have last 150 tokens of this 1000 words as well. to secure the data 
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    #generate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

    #creating vectorstore - FAISS
    #(chunks and embeddings are given as variables as those are the ones that must be stored in the database)
    vector_store = FAISS.from_texts(chunks, embeddings)

    #get user question
    user_question = st.text_input("Type your question here")

    #do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question) #where is this question stored
        #st.write(match)

        #define the llm (these values are changed according to the use case - parameter optimization)
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0, #lower the value, more specific the model is. ig higher more raqndom information are given
            max_tokens = 1000,
            model_tokens = "gpt-3.5-turbo"
        )

        #output results
        chain = load_qa_chain(llm, chain_type = "stuff")
        response = chain.run(input_documents= match, question = user_question)
        st.write(response)

