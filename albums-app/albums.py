from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

llm = ChatOpenAI(temperature=0.6, model='gpt-3.5-turbo')

def count_albums(artist):
    
    prompt_count_albums = PromptTemplate(
        input_variables=['artist'],
        template="Quantos álbuns {artist} já lançou em sua carreira?"
    )
    count_albums_chain = LLMChain(llm=llm, prompt=prompt_count_albums)

    response = count_albums_chain.invoke({'artist': artist})

    return response['text']

def count_albums_rag(artist, url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    embeddings = OpenAIEmbeddings()
    documents = RecursiveCharacterTextSplitter().split_documents(docs)

    vectordb = FAISS.from_documents(documents, embeddings)
    retriever = vectordb.as_retriever()

    prompt = ChatPromptTemplate.from_template(""" Responda a pergunta com base apenas no contexto
    {context}
    Pergunta: {input}
    """)

    documents_chain = create_stuff_documents_chain(llm, prompt)
    retriver_chain = create_retrieval_chain(retriever, documents_chain)

    response = retriver_chain.invoke({"input": "Quantos álbuns {artist} já lançou em sua carreira?"})
    return response['answer']

if __name__=="__main__":
    print('=========================================')
    print('Resposta direta do gpt-3.5-turbo:')
    print(count_albums('Taylor Swift'))
    print('=========================================')
    print('Resposta via RAG com o gpt-3.5-turbo:')
    print(count_albums_rag('Taylor Swift', 'https://pt.wikipedia.org/wiki/Discografia_de_Taylor_Swift'))
    print('=========================================')
