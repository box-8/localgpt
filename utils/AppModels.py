import time
import streamlit as st
import os
import socket
import subprocess
import psutil
import time

from utils.embeddings import * 
from utils.session import BasicSession

script_directory = os.path.abspath(os.path.dirname(__file__))
# Chemin absolu vers le répertoire "models"
models_directory = os.path.abspath(os.path.join(script_directory, "..", "models")).replace("\\", "/")
# Chemin absolu vers le répertoire de l'environnement virtuel (.venv/Scripts/activate.bat)
venv_directory = os.path.abspath(os.path.join(models_directory, "..", ".venv", "Scripts", "activate.bat")).replace("\\", "/")


class AppModels(BasicSession):
    def __init__(self) -> None:
        self.session_init()
    
    # build user interface to see or change the LLM in use  
    def ui_panel(self):
        st.markdown(f"## Gestion des modèles 🍀 {st.session_state.llm_modelname}")
        
        # CACHED_LLM = []
        files = self.list_models()
        if not files:
            st.write("Aucun fichier trouvé dans le répertoire 'models'")
        else:
            
            line = st.empty()
            col1, col2, col3 = line.columns(3)
            
            with col1 :
                if st.session_state.llm_port == "groq":
                    btype = "primary"
                else:
                    btype = "secondary"
                button_groq = st.button(f"Mistral 8x7b on GROQ", type=btype, key=f"port_groq", on_click=self.set_llm_service, args=("groq mistral","groq",))
            with col2 :
                button_groq2 = st.button(f"Démarrer le modèle sur Groq", type="secondary", key=f"port_groq2", on_click=self.set_llm_service, args=("groq mistral","groq",))
                
            for file in files:
                port_number = self.generate_port_number(file)
                with col1:
                    if st.session_state.llm_port == port_number:
                        btype = "primary"
                        is_active = True
                    else : 
                        btype = "secondary"
                        is_active = False
                    
                    button_set = st.button(f"{file}", type=btype, key=f"port_{port_number}", on_click=self.set_llm_service, args=(file,port_number,))
                        
                with col2:
                    container = st.empty()
                    
                    if self.is_service_running(port_number):
                        self.button_stop(container, port_number)
                        is_running = False
                    else:
                        self.button_start(container, file, port_number)
                        is_running = True
                
                
                # v1.2 : cache local LLM
                # entry = {"name":file, "port": port_number, "is_running": is_running, "is_active":is_active}
                # CACHED_LLM.append(entry)
            if st.session_state.llm_modelname =="": 
                self.no_llm_warn()
               
    def is_service_running(self,port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    
    def list_models(self):
        files = os.listdir(models_directory)
        return [f for f in files if os.path.isfile(os.path.join(models_directory, f))]

    def stop_llm_service(self, port):
        for proc in psutil.process_iter(['pid', 'net_connections']):
            try:
                for conn in proc.net_connections():
                    if conn.laddr.port == port:
                        proc.terminate()
                        return proc
            except psutil.NoSuchProcess:
                # Ignore les processus qui n'existent plus
                pass
        return "Not Stopped"


    

    # génére le bouton pour démarrer un LLM 
    def button_start(self, container, file, port_number):
        model_path = os.path.join(models_directory, file)
        start_button = container.button(f"Démarrer le modèle sur {port_number}", key=file)
        if start_button:
            is_started, message = self.start_llm_service(model_path=model_path,port=port_number)
            st.write(message)
            self.set_llm_service(file, port_number)
            self.button_stop(container,port_number)    
    
    # génére le bouton pour arrêter un LLM 
    def button_stop(self, container,port_number):
        test_stoped= container.button(f"Arrêter le modèle sur {port_number}", on_click=self.stop_llm_service, args=(port_number,))
        if not test_stoped :
            pass
        elif test_stoped =="Not Stopped":
            st.error(f"Server could not be stopped {port_number}")
        else:
            st.write(test_stoped) 
            st.success(f"Server stopped {port_number}")
    
    # fonction qui fait 
    def set_llm_service(self, modelname, port):
        st.session_state.llm_modelname = modelname
        st.session_state.llm_port = port
    
    # démarre le LLM lorsque l'on clique sur le bouton
    def old_start_llm_service(self,model_path, port):
        # Commande à exécuter
        command = ["python", "-m", "llama_cpp.server", "--model", model_path, "--port", str(port)]

        # Exécute la commande en activant d'abord l'environnement virtuel
        # process = subprocess.Popen([venv_directory, "&&"] + command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # Exécute la commande en activant d'abord l'environnement virtuel et en redirigeant les E/S
        process = subprocess.Popen([venv_directory, "&&"] + command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        # Attendre que le processus se ter mine et récupérer la sortie
        return True
    
    

    def start_llm_service(self, model_path, port):
        # Vérifier si le chemin du modèle existe
        if not os.path.exists(model_path):
            return False, f"Model path '{model_path}' does not exist."

        # Commande sous forme de liste d'arguments
        command = ["llama_cpp.server", "--model", model_path, "--port", str(port)]

        try:
            # Exécution de la commande sans shell=True
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Attendre un court délai pour que le serveur démarre
            time.sleep(2)  # Attendre 2 secondes

            # Vérifier si le processus est toujours en cours d'exécution
            if process.poll() is None:
                return True, "Service started successfully"
            else:
                # Lire et retourner l'erreur depuis stderr avec gestion de l'encodage
                try:
                    stderr_output = process.stderr.read().decode('utf-8')  # Tenter de décoder en UTF-8
                except UnicodeDecodeError:
                    stderr_output = process.stderr.read().decode('cp1252')  # Décoder en cp1252 si UTF-8 échoue
                
                return False, f"Service failed to start. Error: {stderr_output}"

        except FileNotFoundError:
            return False, "llama_cpp.server not found. Make sure it's installed and in your PATH."
        except Exception as e:
            return False, f"An unexpected error occurred: {str(e)}"

    
    # définit un numéro de port pour chaque modèle 
    def generate_port_number(self,file_name):
        # Convertir le nom de fichier en une valeur numérique en utilisant la somme des codes ASCII des caractères
        file_value = sum(ord(char) for char in file_name)
        # Utiliser le modulo pour obtenir un numéro de port entre 1500 et 1600
        port_number = 1500 + (file_value % 101)  # 101 est un nombre premier pour une distribution plus uniforme
        return port_number

    
    
    
    
                
                
                
if __name__ == "__main__":
    App = AppModels()
    App.main()



#############################################################################
# todo v1.2
# implémenter LLM cached local services
#############################################################################
    # def cached_llm(self):
    #     CACHED_LLM = []
    #     files = self.list_models()
    #     if not files:
    #         st.write("Aucun fichier trouvé dans le répertoire 'models'")
    #     else:
    #         for file in files:
    #             port_number = self.generate_port_number(file)
    #             if st.session_state.llm_port == port_number:
    #                 is_active = True
    #             else : 
    #                 is_active = False
    #             if self.is_service_running(port_number):
    #                 is_running = True
    #             else:
    #                 is_running = False
    #             entry = {"name":os.path.basename(file), "port": port_number, "is_running": is_running, "is_active":is_active}
    #             CACHED_LLM.append(entry)
    #     return CACHED_LLM
    
    
    # # todo v1.2
    # def models_dropdown(self,container=None):
    #     options = list()
    #     cachedllms = self.cached_llm()
    #     idex = 0 
    #     for index, models in enumerate(cachedllms):
    #         name = models["name"]
    #         port_number = models["port"]
    #         if str(st.session_state.llm_port) == str(port_number):
    #             idex = index
    #         if models["is_running"] :
    #             prefix = "👍"  
    #         else:
    #             prefix = "👎"
                
    #         if models["is_active"] :
    #             prefix1 = "✔️" 
    #         else:
    #             prefix1 = " "
            
    #         entry = f"{prefix} {prefix1} : {name} :{port_number}"
    #         options.append(entry)
        
    #     if not container : 
    #         selected_entry = st.selectbox("llm model", options=options, index=idex)
    #     else : 
    #         selected_entry = container.selectbox("llm model", options=options, index=idex)
    #     if selected_entry:
    #         parts = selected_entry.split(':')
    #         selected_port = int(parts[-1].strip())
    #         selected_file = parts[-2].strip()
    #         model_path = os.path.join(models_directory, selected_file)
    #         st.session_state.llm_port = selected_port
    #         if not self.is_service_running(selected_port) :
    #             is_starteed, message = self.start_llm_service_n(model_path=model_path,port=selected_port)
    #             st.write(message)