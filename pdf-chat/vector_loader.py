#import Essential dependencies

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

#create a new file named vectorstore in your current directory.
if __name__=="__main__":
        DB_FAISS_PATH = 'pdf-chat/vectorstore/db_faiss'
        loader=PyPDFLoader("./pdf-chat/CompBrasil_39_180.pdf")
        docs=loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        vectorstore.save_local(DB_FAISS_PATH)
