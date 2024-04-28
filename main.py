import os
from dotenv import load_dotenv
from groq import Groq

import streamlit as st

load_dotenv()

st.title("MSE Expert")
input_meesage = st.chat_input("Ask from expert...")


client = Groq(
    api_key=os.getenv("groq_API"),
)

if input_meesage:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": str(input_meesage),
            }
        ],
        model="mixtral-8x7b-32768",
    )

    output = chat_completion.choices[0].message.content
    st.write(output)
