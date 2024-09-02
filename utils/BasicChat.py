import os
import streamlit as st
import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyMuPDFLoader 

from utils.embeddings import EMBEDDINGS
from utils.BasicSession import BasicSession
from utils.BasicLLM import BasicLLM


class BasicChat(BasicSession, BasicLLM):
    def _init(self, title="", icon=""):
        self.session_init()
        self.title = title
        self.icon = icon
        st.set_page_config(page_title=self.title, page_icon=self.icon, layout='wide')
        st.title(self.title + " "+ self.icon)
        
        self.llmLocal(st.session_state.llm_port) 
    
    
    
    # créer un attribut dans le chunck ou le met à jour
    def set_attr(self,chunk, attrName, attrValue):
        if hasattr(chunk, 'metadata'):
            chunk.metadata[attrName] = attrValue
        else:
            chunk.metadata = {attrName: attrValue}
            
    # vectorise un document
    def vectoriser(self, file_path, collectionName="langchain"):
        loader = PyMuPDFLoader(file_path=file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            #add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(collectionName)
        # Ajouter le nom de la collection aux métadonnées de chaque chunk
        for chunk in chunks:
            self.set_attr(chunk=chunk, attrName="collectionName",attrValue=collectionName)
            self.set_attr(chunk=chunk, attrName="filename",attrValue=os.path.basename(file_path)
)
            # if hasattr(chunk, 'metadata'):
            #     chunk.metadata['collectionName'] = collectionName
            # else:
            #     chunk.metadata = {'collectionName': collectionName}

            os.path.basename(file_path)
        """
        Vectorisation en cours
        """
        self.chroma_db.add_documents(documents=chunks)
        st.sidebar.warning(f"Le document a été vectorisé avec succès à l'emplacement : {file_path}")
        st.balloons()
        st.experimental_rerun()
           
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
