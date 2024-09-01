import streamlit as st
import time
import chromadb
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from utils.embeddings import EMBEDDINGS
from utils.session import BasicSession
from utils.llm import BasicLLM
from utils.models import AppModels

class BasicChat(BasicSession, BasicLLM):
    def _init(self, title="", icon=""):
        self.session_init()
        self.title = title
        self.icon = icon
        st.set_page_config(page_title=self.title, page_icon=self.icon, layout='wide')
        st.title(self.title + " "+ self.icon)
        
        self.llmLocal(st.session_state.llm_port) 
        
    def ui_context(self):
        opt_system_context = self.container_options.text_area("Contexte du Système :",key="opt_system_context", value=st.session_state.opt_system_context)
        if opt_system_context:
            self.setContext(context=opt_system_context)
        
    def setContext(self, context):
        newcontext = SystemMessage(content=context)
        st.session_state.history[0] = newcontext
        return newcontext
            
    def getDebug(self, text=""):
        if text =="":
            phrase = """Victor Hugo était un poète, romancier et dramaturge français, né le 26 février 1802 et décédé le 22 mai 1885. Il est considéré comme l'un des plus grands écrivains français de tous les temps. Ses œuvres les plus célèbres comprennent "Les Misérables" et "Les Contes du temps passé". Hugo a été également un député et sénateur de la République française, ainsi qu'un partisan actif des idées républicaines"""
        else : 
            phrase = text
            # a@todo v1.2
            # AppModels().models_dropdown()
        mots = phrase.split()
        for mot in mots:
            yield mot + " "
            time.sleep(0.1)
    
    def ui_k(self):
        opt_kfragments = self.container_options.slider('Retrieved fragments', value=st.session_state.opt_kfragments, min_value=2,max_value=6)
        if opt_kfragments:
            st.session_state.opt_kfragments = opt_kfragments
    
    def ui_debug(self):
        opt_debug = self.container_options.toggle('Debug mode', key="opt_debug_selector", value=st.session_state.opt_debug)
        st.session_state.opt_debug = opt_debug
    
    
    # Affichage de l'onglet Options soit dans la page home, soit dans le dropdown
    def options(self,container=None):
        # Si l'utilisateur choisit d'afficher le texte, afficher le text_area
        if not container:
            self.no_llm_warn()
            self.container_options = st.popover("options")
            warn = "Choisir un modèle"
            st.page_link("home.py",label=warn)
            
        
        else:
            
            self.container_options.info(f"{st.session_state.llm_modelname} running on port : {st.session_state.llm_port} ")
        
        # affichage du contexte des conversations
        self.ui_context()
        col1, col2 = self.container_options.columns(2)
        with col1:
            self.ui_debug()
                
            if st.button("show session"):
                self.session_show()
            
        with col2:
            if self.container_options.button("Effacer chat"):
                self.history_reset()
            if st.button("kill session"):
                self.session_kill()
            self.ui_k()
        
        # @todo V1.2 
        # AppModels().models_dropdown(self.container_options)
    
    # affichage de l'onglet chat simple
    def chat(self):
        # conversation
        for message in st.session_state.history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)

        # user input
        user_query = st.chat_input("Type your message here...")
        if user_query is not None and user_query != "":
            st.session_state.history.append(HumanMessage(content=user_query))
            with st.chat_message("Human"):
                st.markdown(user_query)
            with st.chat_message("AI"):
                try:
                    response = st.write_stream(self.ask(user_query))
                except:
                    response = st.write_stream(self.getDebug("Error connection to llm server"))
            st.session_state.history.append(AIMessage(content=response))

    # fonction générique pour poser une question au LLM actif 
    def ask(self, query) :
        if st.session_state.opt_debug :
            return self.getDebug()
        else:
            return self.get_response(query)
        
    # fonction générique pour recevoir la réponse du  LLM actif 
    # tofo implementer l'usage d'une question non inscrite dans l'historique de conversation
    def get_response(self, query):
        chatactual = st.session_state.history
        prompt = ChatPromptTemplate.from_messages(chatactual)
        chain = prompt | self.llm | StrOutputParser()
        return chain.stream({})
