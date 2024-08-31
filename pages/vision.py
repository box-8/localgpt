import streamlit as st
import base64
from utils.ui import BasicChat
from openai import OpenAI

class AppVision(BasicChat):
  
    def __init__(self) -> None:
        # Initialisation de la classe de base et de l'API locale
        super()._init("Vision", "👀")
        self.base_url = "http://localhost:1552/v1"  # Base URL pour le LLM local
        # Point to the local server
        self.llm = OpenAI(base_url=self.base_url, api_key="not-needed")
        
    
    def main(self):
        # Interface utilisateur pour télécharger une image
        self.uploaded_doc = st.file_uploader("Télécharger une image", type=["jpg", "png", "bmp", "jpeg"])
        if self.uploaded_doc is not None:
            # Si un fichier est uploadé, passer à la discussion
            self.chat()

    
    
    def ask(self, query):
        if self.uploaded_doc is not None:
            try:
                # Encodage de l'image en base64
                image = self.uploaded_doc.read()
                self.base64_image = base64.b64encode(image).decode("utf-8")
                # Envoyer la requête au modèle avec l'image et la question
                return self.get_response(query)
            except Exception as e:
                return f"Erreur lors de la lecture de l'image: {e}"
        else:
            return "Veuillez télécharger une image avant de poser une question."

    def get_response(self, query=""):
        # Génération de la requête avec le modèle
        if query =="":
            return ""
        try:
            # Création du payload pour l'API LLM local
                        
            completion = self.llm.chat.completions.create(
            model="local-model", # not used
            messages=[
                {
                "role": "system",
                "content": "Vous êtes un ingénieur en construction qui analyse des photos de chantier.",
                },
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{query}"},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.base64_image}"
                    },
                    },
                ],
                }
            ],
            max_tokens=1000,
            stream=True
            )
            #st.session_state.history.append(HumanMessage(content=user_query))
            response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    print(f"piece of response {chunk}")
                    yield chunk
        except Exception as e:
            return f"Erreur lors de l'envoi de la requête: {e}"

    
    
if __name__ == "__main__":
    app = AppVision()
    app.main()