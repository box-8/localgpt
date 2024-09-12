import os
import streamlit as st
import pandas as pd
from langchain.vectorstores.chroma import Chroma
from utils.AppModels import AppModels
from utils.embeddings import DATA_PATH, EMBEDDINGS
from utils.AppRag import AppRag





class AppOfferAnalysis(AppRag):

    def std_uploader(self, selfDoc, filename, key):
        
        file_rename=f"{self.collectionName}_{filename}"
        file_rename_path = os.path.join(DATA_PATH, file_rename)
        st.write(key.upper() )
        
        
        if os.path.exists(file_rename_path):
            
            colf1, colf2, colf3 = st.columns(3)
            with colf1:
                button = st.button(f"voir le pdf",key=file_rename)
                if button: 
                    os.startfile(file_rename_path)
           
                button1 = st.button(f"supprimer ",key="delete_"+file_rename)
                if button1: 
                    os.remove(file_rename_path)
                    st.sidebar.success(f"Le document {file_rename} a été supprimée avec succès.")
                    selfDoc = None
                    st.experimental_rerun()
            with colf2:
                button2 = st.button(f"afficher chroma ",key="vecteur_"+file_rename)
                if button2: 
                    self.query_chroma(self.collectionName)
        else:
            selfDoc = st.file_uploader(label="uploader le fichier", accept_multiple_files=False, key=key)
            if selfDoc is not None :
                saved_path = self.save_uploaded_doc(uploaded_file=selfDoc, file_rename=file_rename )
                self.vectorise_document(file_path=saved_path, collectionName=self.collectionName)
                st.balloons()
                st.experimental_rerun()
        

    def run_app(self) :
        self.CCTP = None
        self.Offre1 = None
        self.Offre2 = None
        self.Offre3 = None
        super().sidebar()
        tab1, tab2, tab3, tab4 = st.tabs(["CCTP", "Offres", "Analyse", "Options"])
            
        with tab1:
            st.header(f"Cahier des charges #{self.collectionName}#")
            self.std_uploader(selfDoc=self.CCTP, filename="CCTP.pdf", key="cctp")
            
        with tab2:
            st.header(f"Dossier Offres")
            colf1, colf2, colf3 = st.columns(3)
            with colf1:
                self.std_uploader(selfDoc=self.Offre1, filename="Offre1.pdf", key="offre1")
            with colf2:
                self.std_uploader(selfDoc=self.Offre2,  filename="Offre2.pdf", key="offre2")
            with colf3:
                self.std_uploader(selfDoc=self.Offre3,  filename="Offre3.pdf", key="offre3")

        with tab3:
            # Liste des fichiers du répertoire "data"
            st.header(f"Analyse des offres")
            button1 = st.button('Analyser')
            if button1:
                st.toast(f"L'analyse est en cours ...")
            st.session_state['container'] = st.container()
            st.session_state['container'].write("default ...")
                
        with tab4:
            models = AppModels()
            models.ui_panel()
        
        if 'container' not in st.session_state:
            st.session_state['container'] = "..."
        
        collection = self.client.get_or_create_collection(name=self.collectionName)
        df = pd.DataFrame.from_dict(collection.get())
        # manque selection sur docname
        write = st.dataframe(df, width=1400)
        st.container().write(write)
        
    def query_chroma(self,filename=""): 
        
        """
        Fonction pour récupérer et filtrer les documents de la collection Chroma
        selon la metadata file_name.
        """
        # Récupération de la collection depuis Chroma
        collection = self.client.get_or_create_collection(name=self.collectionName)
        
        # Récupérer toutes les données de la collection
        data = collection.get()
        
        # Extraire les métadonnées des documents
        metadatas = data['metadatas']
        
        # Filtrer les documents par filename
        filtered_data = [meta for meta in metadatas if meta.get('filename') == filename]
        
        # Créer un DataFrame avec les métadonnées filtrées
        df = pd.DataFrame(filtered_data)
        
        return df
        

        

    def delete_document(self, collectionName, documentName, client):
        collection = self.client.get_or_create_collection(name=collectionName)
        df = pd.DataFrame.from_dict(collection.get())
        filtered_ids = set()
        for index, row in df.iterrows():
            metadatas = row['metadatas']
            if documentName in metadatas["file_path"]:
                filtered_ids.add(row['ids'])

        chroma_db = Chroma(
            client=client,
            collection_name=collectionName,
            embedding_function=EMBEDDINGS(),
        )
        for document_id in filtered_ids:
            chroma_db.delete(document_id)

        # Persist the changes
        # chroma_db.persist()

        return collection

    def create_collection(self,name):
        collection = self.client.get_or_create_collection(name=name)
        self.init_collections()
        # st.session_state.collection = self.noms_collections[0]
        "collection created"
        return collection

    def rename_collection(self, name):
        if name =="":
            return
        else :
            collection = self.client.get_or_create_collection(name=st.session_state.collection)
            collection.modify(name)

    def delete_collection(self,name):
        collection = self.client.delete_collection(name=name)
        self.init_collections()
        "collection supprimée"
        return collection


if __name__ == "__main__":
    app = AppOfferAnalysis("Rapport d'Analyse","💵")
    app.run_app()