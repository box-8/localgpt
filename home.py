# chroma run --path ./chroma 
# streamlit run home.py

import os
import streamlit as st
import pandas as pd
from langchain.vectorstores.chroma import Chroma
from utils.AppModels import AppModels
from utils.embeddings import DATA_PATH, EMBEDDINGS
from utils.AppRag import AppRag


class AppHome(AppRag):
    
    def run_app(self) :
        _Collections = self.client.list_collections()  # obtenir la liste des _Collections
        if not _Collections:  # vérifier si la liste des _Collections est vide
            st.write("Il n'y a pas de collections.")
            self.create_collection("langchain")
        
        super().sidebar()
        collection_selectionnee = self.client.get_collection(self.collectionName)
                
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Collection", "Documents", "Fichiers", "Models", "Options"])
        with tab1:
            st.header(f"Collection #{self.collectionName} : management")
            new_collection_name = st.text_input(label="",placeholder="Nom de la collection", value=self.collectionName)
        
            colc1, colc2, colc3, colc4 = st.columns(4)
            with colc4 :
                if st.button("⚠️ Supprimer la collection", key="delete_collection") :
                    self.delete_collection(self.collectionName)
                    st.rerun()
            with colc2 :

                if st.button("📓 Nouvelle collection ...", key="collection_create"): 
                    if new_collection_name == self.collectionName :
                        st.toast("Le nom choisi est identique. Choisir un nouveau nom ...")
                    elif new_collection_name in [collection.name for collection in _Collections]:
                        st.toast("Une collection avec ce nom existe déjà,\nchoisir un autre nom")
                    else:
                        collection_selectionnee = self.create_collection(name=new_collection_name)
                        "collection créée"  
                        st.rerun() 
            with colc3: 
                if st.button("⚠️ Vider la collection"):
                    self.empty_collection()
                    st.experimental_rerun()

            with colc1:
                if st.button("❓ Rename collection ...", key="collection_rename"): 
                    if new_collection_name == self.collectionName :
                        st.toast("Le nom choisi est identique. Choisir un nouveau nom ...")
                    elif new_collection_name in [collection.name for collection in _Collections]:
                        st.toast("Une collection avec ce nom existe déjà,\nchoisir un autre nom")
                    else:
                        collection_selectionnee = self.create_collection(name=new_collection_name)
                        collection_selectionnee = self.rename_collection(name=new_collection_name)
                        st.toast(f"Collection renommée {new_collection_name}")    
                        st.rerun() 
                        
            self.read_collection(collection_selectionnee)
            
            
        with tab2:
            st.header(f"Collection #{self.collectionName} : documents")
            colf1, colf2= st.columns(2)
            with colf1:
                uploaded_doc = st.file_uploader(f"🚩 Vectoriser un document dans {self.collectionName} ", type=["pdf", "txt", "docx"])
                if uploaded_doc is not None:
                    if self.collectionName == "" :
                        st.toast("Choisir une collection dans laquelle vectoriser le document")
                    else:    
                        file_path = self.save_uploaded_doc(uploaded_doc, file_rename= uploaded_doc.name.lower())
                        self.vectorise_document(file_path=file_path, collectionName=self.collectionName)
                        #self.read_collection(collection_selectionnee)
                        

                # docs_in_collection = self.list_documents(collectionName=self.collectionName)
                # nom_document_supprimer = st.selectbox("Choisissez le document à supprimer de la collection", self.docs_in_collection)
                # if st.button("⚠️ Supprimer un document",key="delete_document_confirm"):
                #     self.delete_document(collectionName=self.collectionName, documentName=nom_document_supprimer)

                
            with colf2:
                if st.button("⚠️ Supprimer les documents",key="multi_delete_document_confirm"):
                    for selected_document in self.selected_docs:
                        self.delete_document(collectionName=self.collectionName, documentName=selected_document)
                
                     
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
                    self.vectorise_document(file_path=selected_data_path, collectionName=self.collectionName)
        with tab4:
            st.markdown(f"## Gestion des modèles 🍀 {st.session_state.llm_modelname}")
            buttonModels = st.button('Choisir le modèle')
            if buttonModels :
                models = AppModels()
                models.ui_panel()
        with tab5: 
            st.header("Options")
            self.container_options = st.container()
            super().options(self.container_options)
            
    def ui_actions(self,collection_selectionnee):
        options_menu = ["Choisir une action","Ajouter un élément", "Mettre à jour un élément", "Supprimer un élément"]
        option_selectionnee = st.selectbox("Choisissez une action", options_menu)

        if option_selectionnee == "Ajouter un élément":
            nouvelle_embedding = st.text_input("Entrez l'embedding du nouvel élément :")
            nouveau_metadata = st.text_input("Entrez les métadonnées du nouvel élément :")
            nouvel_id = st.text_input("Entrez l'ID du nouvel élément :")

            # Ajouter le nouvel élément à la collection
            if st.button("Ajouter l'élément"):
                collection_selectionnee.add(embeddings=[float(nouvelle_embedding)],
                                metadatas={"meta": nouveau_metadata},
                                ids=nouvel_id)
                st.write(f"L'élément {nouvel_id} a été ajouté.")

        elif option_selectionnee == "Mettre à jour un élément":
            # Mettre à jour un élément existant dans la collection
            id_element_a_mettre_a_jour = st.text_input("Entrez l'ID de l'élément à mettre à jour :")
            embedding_mis_a_jour = st.text_input("Entrez l'embedding mis à jour de l'élément :")
            metadata_mis_a_jour = st.text_input("Entrez les métadonnées mises à jour de l'élément :")
            if st.button("Mettre à jour l'élément"):
                collection_selectionnee.update(ids=id_element_a_mettre_a_jour,
                                    embeddings=[float(embedding_mis_a_jour)],
                                    metadatas={"meta": metadata_mis_a_jour})
                st.write(f"L'élément {id_element_a_mettre_a_jour} a été mis à jour.")

        elif option_selectionnee == "Supprimer un élément":
            # Supprimer un élément de la collection
            id_element_a_supprimer = st.text_input("Entrez l'ID de l'élément à supprimer :")
            if st.button("Supprimer l'élément"):
                collection_selectionnee.delete(ids=id_element_a_supprimer)
                st.write(f"L'élément {id_element_a_supprimer} a été supprimé.")
  
    def read_collection(self,collection):
        df = pd.DataFrame.from_dict(collection.get())
        st.dataframe(df, width=1400)
        return df 
        # Sélectionner une ligne en cliquant sur un index dans la colonne 'ids'
        
        selected_id = st.selectbox('Sélectionner une ligne', df['ids'])

        # Afficher les attributs de la ligne sélectionnée
        if selected_id is not None:
            selected_row = df[df['ids'] == selected_id].iloc[0]
            st.write('Attributs de la ligne sélectionnée :')
            for index, value in selected_row.items():
                st.write(f"{index}: {value}")
                
    def delete_document(self, collectionName, documentName):
        collection = self.client.get_or_create_collection(name=collectionName)
        df = pd.DataFrame.from_dict(collection.get())
        filtered_ids = set()
        for index, row in df.iterrows():
            metadatas = row['metadatas']
            if documentName in metadatas["file_path"]:
                filtered_ids.add(row['ids'])
        
        chroma_db = Chroma(
            client=self.client,
            collection_name=collectionName,
            embedding_function=EMBEDDINGS(),
        )
        for document_id in filtered_ids:
            chroma_db.delete(document_id)

        # Persist the changes
        # chroma_db.persist()

        return collection

    # retourne un objet collection
    def create_collection(self,name):
        collection_object = self.client.get_or_create_collection(name=name)
        self.init_collections()
        # st.session_state.collection = self.noms_collections[0]
        "collection created"
        return collection_object
    
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
    app = AppHome("Vector DB","🏝️")
    app.run_app()