import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OpenAI_API")

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI


file = "I am Lasith Dissanayake and I am 24 years old."

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

user_question = input("Ask the question :")

    

if user_question:
    match = vector_store.similarity_search(user_question) 

    llm = ChatOpenAI(
        openai_api_key = OPENAI_API_KEY,
        temperature = 0, 
        max_tokens = 1000
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    chain = user_question | llm

    response = chain.run(input_documents= match, question = user_question)

    print(response)

