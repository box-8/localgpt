####
#### Streamlit Streaming using LM Studio as OpenAI Standin
#### run with `streamlit run app.py`

# !pip install pypdf langchain langchain_openai 


from utils.BasicChat import BasicChat


class AppChat(BasicChat):
    def __init__(self, title="CHAT Bot", icon="🤖"):
        self._init(title,icon)
        
    def main(self):
        # sidebar
        
        self.options()
        self.chat()  
        
    
        
if __name__ == "__main__":
    app = AppChat("Chatbot","🎶")
    app.main()