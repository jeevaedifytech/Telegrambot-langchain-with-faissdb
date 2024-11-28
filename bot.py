import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import logging
import tempfile
import torch 
# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary to store PDF documents per user
user_pdf_data = {}

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return None
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store from text chunks
def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # "hkunlp/instructor-xl"
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to set up conversational chain
def get_conversation_chain(vectorstore):
    #llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.5, "max_length":512})   # can use google/flan-t5-xxl
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Handle user messages
def handle_message(update: Update, context: CallbackContext):
    user_question = update.message.text
    user_id = update.message.from_user.id
    
    if user_id in user_pdf_data and 'conversation' in user_pdf_data[user_id]:
        try:
            response = user_pdf_data[user_id]['conversation']({'question': user_question})
            chat_history = response['chat_history']
            bot_reply = chat_history[-1].content if chat_history else "Sorry, I couldn't find an answer."
            update.message.reply_text(bot_reply)
        except Exception as e:
            update.message.reply_text(f"Sorry, there was an error processing your request: {str(e)}")
            logger.error(f"Error handling message: {e}")
    else:
        update.message.reply_text("Please upload a PDF first to start interacting with the bot.")

# Handle PDF file uploads
def handle_pdf(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id

    if update.message.document:
        try:
            file = update.message.document.get_file()
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                file.download(temp_file.name)

                if user_id not in user_pdf_data:
                    user_pdf_data[user_id] = {'pdfs': [], 'conversation': None}
                
                user_pdf_data[user_id]['pdfs'].append(temp_file.name)

                # Get the text from the uploaded PDF
                raw_text = get_pdf_text(user_pdf_data[user_id]['pdfs'])

                if not raw_text:
                    update.message.reply_text("Sorry, there was an issue extracting text from the PDF.")
                    return

                # Split text into chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store and conversation chain
                vectorstore = get_vectorstore(text_chunks)
                user_pdf_data[user_id]['conversation'] = get_conversation_chain(vectorstore)

                update.message.reply_text("Your document has been processed. You can now ask me questions!")
        except Exception as e:
            update.message.reply_text(f"Error uploading PDF: {str(e)}")
            logger.error(f"Error handling PDF upload: {e}")
    else:
        update.message.reply_text("Please send a PDF document.")

# Start command for the bot
def start(update: Update, context: CallbackContext):
    update.message.reply_text("Welcome to the PDF Chatbot! Please upload a PDF document to get started.")

# Main function to run the bot
def main():
    # Load the Telegram API token from environment variables
    telegram_api_token = os.getenv('TELEGRAM_API_TOKEN')

    if not telegram_api_token:
        logger.error("TELEGRAM_API_TOKEN is missing from environment variables.")
        return

    # Set up the Updater with the bot API token
    updater = Updater(telegram_api_token, use_context=True)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Add command and message handlers
    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    dispatcher.add_handler(MessageHandler(Filters.document.mime_type('application/pdf'), handle_pdf))

    # Start the Bot
    updater.start_polling()
    updater.idle()

    
if __name__ == '__main__':
    main()
    print(torch.cuda.is_available())