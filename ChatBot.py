import os
import sys
sys.path.append('../..')
import apikey
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import tkinter as tk
from tkinter import scrolledtext

os.environ["OPENAI_API_KEY"] = apikey.APIKEY

pdf_files = [
    r"C:\Users\krish\Desktop\Python VS\newenv\ChatbotZIP\data_resumes\associate-data-scientist-resume-example.pdf",
    r"C:\Users\krish\Desktop\Python VS\newenv\ChatbotZIP\data_resumes\data-scientist-intern-resume-example.pdf",
    r"C:\Users\krish\Desktop\Python VS\newenv\ChatbotZIP\data_resumes\data-scientist-resume-example.pdf",
    r"C:\Users\krish\Desktop\Python VS\newenv\ChatbotZIP\data_resumes\entry-level-data-scientist-resume-example.pdf",
    r"C:\Users\krish\Desktop\Python VS\newenv\ChatbotZIP\data_resumes\senior-data-scientist-resume-example.pdf"
]

all_documents = []

# Data Loading
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    all_documents.extend(documents)

# Document splitting

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=10
)
texts = text_splitter.split_documents(all_documents)

# Data storing

persist_directory = "./storage"
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embeddings,
                                 persist_directory=persist_directory)
vectordb.persist()

#Retrieval

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

#QnA

def handle_query():
    user_input = user_input_text.get("1.0", tk.END).strip()
    if user_input.lower() == "exit":
        root.quit()
        return
    query = f"###Prompt {user_input}"
    try:
        llm_response = qa(query)
        response_text.insert(tk.END, llm_response["result"] + "\n\n")
    except Exception as err:
        response_text.insert(tk.END, f"Exception occurred: {str(err)}\n\n")

#UI Interface

root = tk.Tk()
root.title("HR Resume Bot")
root.configure(bg="#4169E1")
user_input_text = scrolledtext.ScrolledText(root, height=5, width=50)
user_input_text.pack(padx=10, pady=50)
submit_button = tk.Button(root, text="Submit", command=handle_query,height=2, width=8)
submit_button.pack(padx=10, pady=15)
initial_response = "Hello! How can I assist you today?"
response_text = scrolledtext.ScrolledText(root, height=11, width=65)
response_text.pack(padx=10, pady=50)
response_text.insert(tk.END, initial_response)
root.mainloop()