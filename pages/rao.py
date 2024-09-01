import os
import streamlit as st
import pandas as pd


from langchain.vectorstores.chroma import Chroma


from utils.AppModels import AppModels
from utils.embeddings import DATA_PATH, EMBEDDINGS
from utils.AppRag import AppRag


class AppOfferAnalysis(AppRag):

    def std_uploader(self, selfDoc,filename, key):
        
        file_rename=f"{self.collectionName}_{filename}"
        file_rename_path = os.path.join(DATA_PATH, file_rename)
    
        if os.path.exists(file_rename_path):
            st.write("popo")
            button = st.button(f"voir {file_rename_path}",key=file_rename)
            if button: 
                os.startfile(file_rename_path)
        else:
            
            # print(f"Le fichier {file_rename} n'existe pas.")
            selfDoc = st.file_uploader(label="", accept_multiple_files=False, key=key)
            saved_path = self.save_uploaded_doc(uploaded_file=selfDoc, file_rename=file_rename )
            self.vectoriser(file_path=saved_path, collectionName=self.collectionName)


    def run_app(self) :
        self.CCTP = None
        self.Offre1 = None
        self.Offre2 = None
        self.Offre3 = None
        super().sidebar()
        tab1, tab2, tab3, tab4 = st.tabs(["CCTP", "Offres", "Analyse", "Options"])

        with tab1:
            st.header(f"Cahier des charges {self.collectionName}")
            
            self.std_uploader(selfDoc = self.CCTP, filename="CCTP.pdf", key="cctp")
            st.write("télécharger le cahier des charges")
        with tab2:
            st.header(f"Dossier Offres")
            colf1, colf2, colf3 = st.columns(3)
            with colf1:
                self.std_uploader(selfDoc =self.Offre1, filename="Offre1.pdf", key="offre1")
            with colf2:
                self.std_uploader(selfDoc =self.Offre2, filename="Offre2.pdf", key="offre2")
            with colf3:
                self.std_uploader(selfDoc =self.Offre3, filename="Offre3.pdf", key="offre3")

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