pip install python-telegram-bot python-dotenv PyPDF2 langchain faiss-cpu

# embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


 def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

pip install python-telegram-bot python-dotenv PyPDF2 langchain faiss-cpu

  https://chatgpt.com/share/6748894e-8794-800b-82de-4a2e52974008 

  pip install -r requirements.txt   
  pip install huggingface-hub==0.14.1 
pip install python-telegram-bot<19.0 
huggingface-cli login  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu




.env file is 
TELEGRAM_API_TOKEN="7995379796:AAF_Imy8fPlh-gWUNXkckY5nexttTlSdpGA1"
HUGGINGFACEHUB_API_TOKEN="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1"
