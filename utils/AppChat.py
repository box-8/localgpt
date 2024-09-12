from utils.BasicChat import BasicChat

class AppChat(BasicChat):
    def __init__(self, title="CHAT Bot", icon="🤖"):
        self._init(title,icon)
        
    def main(self):
        self.options()
        self.chat()  
        
if __name__ == "__main__":
    app = AppChat("Chatbot","🎶")
    app.main()