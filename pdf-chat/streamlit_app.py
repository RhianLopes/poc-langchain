import streamlit as sl
import os 
import sys
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

def load_prompt():
    prompt = """ É necessário responder à pergunta na frase, tal como no conteúdo do pdf. . 
    O contexto e a pergunta do utilizador são apresentados a seguir.
    Contexto = {context}
    Pergunta = {question}
    Se a resposta não estiver no pdf, responda "Não consigo responder a essa pergunta com minha base de informações"
        """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt

def load_llm():
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return llm

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

def extract_data():
    text_chunks = []
    files = filter(lambda f: f.lower().endswith(".pdf"), os.listdir("pdf-chat/uploaded"))
    file_list = list(files)
    for file in file_list:
        loader = PyPDFLoader(os.path.join('pdf-chat/uploaded', file))
        text_chunks += loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
            chunk_size = 512,
            chunk_overlap = 30,
            length_function = len,
            separators= ["\n\n", "\n", ".", " "]
        ))
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings())
    return vectorstore

def initialize_session_state():
    if "knowledge_base" not in sl.session_state:
        sl.session_state["knowledge_base"] = None

def save_uploadedfile(uploadedfile):
    with open(os.path.join("pdf-chat/uploaded", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

def remove_files():
    path = os.path.join(os.getcwd(), 'pdf-chat/uploaded')
    for file_name in os.listdir(path):
        file = os.path.join(path, file_name)
        if os.path.isfile(file) and file.endswith(".pdf"):
            print('Deleting file:', file)
            os.remove(file)

def submit():
  sl.session_state.user_input_prompt = sl.session_state.user_input
  sl.session_state.user_input = ''

if __name__ == '__main__':
    with sl.sidebar:
        with sl.form("my-form", clear_on_submit=True):
            pdf_docs = sl.file_uploader(label="Faça o Upload do seu PDF:", accept_multiple_files=True, type=["pdf"])
            submitted = sl.form_submit_button("Processar")
        if submitted and pdf_docs != []:
            initialize_session_state()
            for pdf in pdf_docs:
                save_uploadedfile(pdf)
            sl.session_state.knowledge_base = extract_data()
            remove_files()
            pdf_docs = []

            alert = sl.success(body=f"Realizado o Upload do PDF com Sucesso!", icon="✅")
            time.sleep(3) 
            alert.empty()

    sl.header("Bem-vindo ao PDF Chat")
    llm=load_llm()
    prompt=load_prompt()
    
    query=sl.text_input(label='Faça uma pergunta sobre o PDF:')
    
    if query:
        try:
            similar_embeddings=sl.session_state.knowledge_base.similarity_search(query)
            similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings())
            
            retriever = similar_embeddings.as_retriever()
            rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
        
            response=rag_chain.invoke(query)
            sl.write(response)
        except:
            alert = sl.warning("Por favor, Realize o Upload do PDF que Deseja Realizar Chat", icon="🚨")
            time.sleep(3)
            alert.empty()
