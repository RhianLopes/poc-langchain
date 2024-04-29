#Import Dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def load_knowledgeBase():
    embeddings=OpenAIEmbeddings()
    DB_FAISS_PATH = 'pdf-chat/vectorstore/db_faiss'
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

#Import Dependencies
from langchain.prompts import ChatPromptTemplate

def load_prompt():
    prompt = """ É necessário responder à pergunta na frase, tal como no conteúdo do pdf. . 
    O contexto e a pergunta do utilizador são apresentados a seguir.
    Contexto = {context}
    Pergunta = {question}
    Se a resposta não estiver no pdf, responda "Não consigo responder a essa pergunta com minha base de informações"
        """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt

#to load the OPENAI LLM
def load_llm():
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return llm

#Import Dependencies
import streamlit as sl
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

if __name__=='__main__':
    sl.header("Bem-vindo ao PDF Chat")
    knowledgeBase=load_knowledgeBase()
    llm=load_llm()
    prompt=load_prompt()
    
    query=sl.text_input('Faça uma pergunta sobre o PDF:')
    
    
    if(query):
        #getting only the chunks that are similar to the query for llm to produce the output
        similar_embeddings=knowledgeBase.similarity_search(query)
        similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings())
        
        #creating the chain for integrating llm,prompt,stroutputparser
        retriever = similar_embeddings.as_retriever()
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
        
        response=rag_chain.invoke(query)
        sl.write(response)

        # python -m streamlit run pdf-chat/streamlit_app.py