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
from utils.chat import BasicChat


class AppRag(BasicChat):
    def __init__(self, title="RAG Bot", icon="🤖"):
        self._init(title,icon)
        
        self.client = chromadb.HttpClient(host='localhost', port=8000)
        self.init_collections()
        
    def init_collections(self):
        self.collections = self.client.list_collections()  # obtenir la liste des collections
        self.noms_collections = [col.name for col in self.collections] # obtenir les noms des collections
        self.collectionName = self.noms_collections[0]
        
    
    def sidebar(self):
        # sidebar
        if not self.collections:  # vérifier si la liste des collections est vide
            st.write("Il n'y a pas de collections.")
            return
        
        self.session_register("collection", self.noms_collections[0])
        
        if st.session_state.collection in self.noms_collections:
            default_idx_collection = self.noms_collections.index(st.session_state.collection)
        else:
            default_idx_collection = 0
        self.collectionName = st.sidebar.selectbox("Choisissez une collection", self.noms_collections, key="collection_selector", index=default_idx_collection)
        self.setCollection(self.collectionName)
        
        self.docs_in_collection = self.list_documents(collectionName = self.collectionName)
        # defaults = self.documents_communs()
        
        self.selected_docs = st.sidebar.multiselect("Choisissez les documents de la collection", self.docs_in_collection, key="selected_docs", default=st.session_state.selected_docs)
    
    def documents_communs(self, tab1, tab2):
        set1 = set(tab1)
        set2 = set(tab2)
        return list(set1.intersection(set2))
    
    def main(self):
        self.options()
        self.sidebar()
        st.write(st.session_state.opt_system_context)
        self.chat()
    
    def list_documents(self, collectionName):
        if collectionName == "Nouvelle collection ...":
            documents = []
        else:
            collection = self.client.get_or_create_collection(name=collectionName)
            df = pd.DataFrame.from_dict(collection.get())
            # Initialisation de la liste des documents unique
            documents = set()
            for index, row in df.iterrows():
                metadatas = row['metadatas']
                documents.add(metadatas["file_path"])
        return documents

    def setCollection(self, name=""):
        if name == "Nouvelle collection ...":
            self.collectionName = name
            st.session_state.collection = name
        else:
            self.collectionName = name
            st.session_state.collection = name
            self.chroma_db = Chroma(
                client=self.client,
                collection_name = self.collectionName,
                embedding_function=EMBEDDINGS(),
            )
    
    def get_response(self, query):
        kelements = st.session_state.opt_kfragments
        st.toast('Gathering information ...', icon='🎉')
        time.sleep(0.5)
        if not self.selected_docs:
            results = self.chroma_db.similarity_search_with_score(query, k=kelements)
        else:
            where = {
                "file_path": {
                    "$in": self.selected_docs
                }
            }
            results = self.chroma_db.similarity_search_with_score(query, filter=where, k=kelements)
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True) 
        context = ""
        for doc, _score in sorted_results:
            context += f"{doc.page_content}"
            page_text = doc.page_content
            pct = str(round(_score, 2))
            mots = page_text.split()
            sumup = " ".join(mots[:100])
            time.sleep(0.5)
            st.toast(f"{pct} % \n"+ sumup, icon = "🍃")
            
           
        template="""
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the Question.
    Question: {question} 
    ------
    Context: 
    {context}  
    ------
    If you don't know the answer, just say that you don't know. 
    Keep the answer concise and always ask the user a relevant question at the end of your reply
    """
        RAGMessage = HumanMessagePromptTemplate.from_template(template)
        chatactual = st.session_state.history.copy() # on crée une copie du chat 
        chatactual.pop() # on retire la question posée par l'utilisateur
        chatactual.append(RAGMessage) # on la remplace par celle calculée RAG
        
        prompt = ChatPromptTemplate.from_messages(chatactual)
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.stream({
            "context": context,
            "question": query,
        })
    


    