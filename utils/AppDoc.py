# Cette classe en capsule les documents pour leur partie physique PDF et leur vectorisation dans la base chromadb
import os


class AppDoc():
    def __init__(self, path):
        self.path = path
        self.filename = os.path.basename()
    
         
        
        