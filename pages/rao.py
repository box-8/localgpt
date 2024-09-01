import os
import streamlit as st
import pandas as pd


from langchain.vectorstores.chroma import Chroma


from utils.AppModels import AppModels
from utils.embeddings import DATA_PATH, EMBEDDINGS
from utils.AppRag import AppRag


class AppOfferAnalysis(AppRag):
    def document_exists(self, collection_name, file_path):
        # Fonction pour vérifier si le document est déjà dans la collection
        collection = self.chroma_db.get_collection(collection_name)
        query_results = collection.query(where={"file_path": file_path})

        # Si des résultats sont trouvés, cela signifie que le document existe déjà
        return len(query_results['documents']) > 0

    def run_app(self) :
        self.CCTP = None
        self.Offre1 = None
        self.Offre2 = None
        self.Offre3 = None
        super().sidebar()
        tab1, tab2, tab3, tab4 = st.tabs(["CCTP", "Offres", "Analyse", "Options"])
        
        with tab1:
            st.header(f"Cahier des charges {self.collectionName}")
            st.write("télécharger le cahier des charges")
            self.CCTP = st.file_uploader(label="choisir le fichier", accept_multiple_files=False)
            if self.CCTP is not None:
                filename = self.CCTP.name.lower()
                saved_path = self.save_uploaded_doc(self.CCTP, file_name = filename)
                self.vectoriser(file_path=saved_path, collectionName="")
            
        with tab2:
            st.header(f"Dossier Offres")
            colf1, colf2, colf3 = st.columns(3)
            with colf1:
                
                
                
                self.Offre1 = st.file_uploader(label="choisir l'offre 1", accept_multiple_files=False)
                if self.Offre1 is not None:
                    filename1 = self.Offre1.name.lower()
                    saved_path = self.save_uploaded_doc(self.Offre1, file_name = filename1)
                    self.vectoriser(file_path=saved_path, collectionName=self.collectionName)
            with colf2:
                self.Offre2 = st.file_uploader(label="choisir l'offre 2", accept_multiple_files=False)
                if self.Offre2 is not None:
                    filename2 = self.Offre2.name.lower()
                    saved_path = self.save_uploaded_doc(self.Offre2, file_name = filename2)
                    self.vectoriser(file_path=saved_path, collectionName=self.collectionName)
            with colf3:
                self.Offre3 = st.file_uploader(label="choisir l'offre 3", accept_multiple_files=False)
                if self.Offre3 is not None:
                    filename3 = self.Offre3.name.lower()
                    saved_path = self.save_uploaded_doc(self.Offre3, file_name = filename3)
                    self.vectoriser(file_path=saved_path, collectionName=self.collectionName)
        
        with tab3:
            # Liste des fichiers du répertoire "data"
            st.header(f"Fichiers sur le disque")
            data_files = os.listdir(DATA_PATH)
            data_files = [file for file in data_files if file.endswith(('.pdf', '.txt', '.docx', '.jpg'))]

            # Afficher la liste des fichiers dans la sidebar
            selected_file = st.selectbox("Choisissez un document parmi les fichiers :", data_files)

            # Charger et afficher l'image sélectionnée depuis le répertoire "images"
            if selected_file:
                selected_data_path = os.path.join(DATA_PATH, selected_file)
                if selected_file.endswith('.jpg'):
                    st.image(selected_data_path, caption=selected_file, use_column_width=True)
            
            col1, col2, col3 = st.columns(3)

            with col1:
                button1 = st.button('Effacer')

            with col2:
                button2 = st.button('Ouvrir')

            with col3:
                button3 = st.button('Vectoriser')
                
            if button1:
                if selected_file:
                    os.remove(selected_data_path)
                    st.sidebar.success(f"Le document {selected_file} a été supprimée avec succès.")
            if button2:
                if selected_file:
                    os.startfile(selected_data_path)
            if button3:
                if selected_file:
                    self.vectoriser(file_path=selected_data_path, collectionName=self.collectionName)
        with tab4:
            models = AppModels()
            models.ui_panel()
       
        
        
    
    
    # sauvegarde un fichier avec le nom spécifié
    def save_uploaded_doc(self, uploaded_file, file_name):
        with open(os.path.join(DATA_PATH, file_name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return os.path.join(DATA_PATH, file_name)
    



    def delete_document(collectionName, documentName, client):
        collection = client.get_or_create_collection(name=collectionName)
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
    app = AppOfferAnalysis("Rapport d'Analyse d'offres","💵")
    app.run_app()