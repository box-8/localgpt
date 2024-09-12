####
#### Streamlit Streaming using LM Studio as OpenAI Standin
#### run with `streamlit run app.py`

from utils.AppChat import AppChat
from utils.AppRag import AppRag
import streamlit as st





chat = AppChat("Chatbot","🎶")
rag = AppRag("RAGbot","🤖") 
    
tab1, tab2, tab3 = st.tabs(["documents", "chat", "options"])

with tab1:
    rag.chat()
    rag.sidebar()
with tab2:
    chat.chat()
with tab3:
    placeholder = st.container()
    rag.container_options = placeholder
    rag.options(rag.container_options)
    
