import faiss
import numpy as np
import spacy
import openai
from dotenv import load_dotenv

from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

# Carregar o modelo de linguagem spaCy
nlp = spacy.load("pt_core_news_sm")

# Corpos de e-mail e suas classificações (0 para suporte técnico, 1 para comercial, 2 para administrativo)
emails = [
    ("Boa tarde, estou tendo dificuldades para resetar minha senha no sistema, o e-mail não chega para mim", 0),
    ("Bom dia, gostaria da segunda via do meu contrato do sistema", 2),
    ("Bom dia, recebi um e-mail sobre o preço do plano premium no valor de R$ 24,90 nos três primeiros meses, gostaria de entender melhor a promoção", 1),
    # Adicione mais e-mails conforme necessário
]

# Vetores de incorporação de palavras para os corpos de e-mail e rótulos de classificação
email_vectors = []
labels = []

# Processamento de texto e criação de vetores
for email, label in emails:
    doc = nlp(email)
    # Calcular a média dos vetores de palavras para representar o corpo do e-mail
    email_vector = np.mean([word.vector for word in doc if word.has_vector], axis=0)
    email_vectors.append(email_vector)
    labels.append(label)

email_vectors = np.array(email_vectors).astype('float32')

# Criar ou carregar um índice FAISS vazio
index = faiss.IndexFlatL2(email_vectors.shape[1])  # Usando a dimensão dos vetores de e-mail como dimensão do índice

# Adicionar vetores ao índice FAISS
index.add(email_vectors)

# Função para realizar uma busca de similaridade no banco de dados FAISS
def search_similar_email(query_vector, k=5):
    # Adicionar uma dimensão extra para tornar o vetor bidimensional
    query_vector = np.expand_dims(query_vector, axis=0)
    _, I = index.search(query_vector, k)
    return I[0]

# Função para classificar o e-mail usando o SDK da OpenAI
def classify_email(emails, context):
    emaill = []
    for i, email in enumerate(emails):
        emaill += f"{i+1}. '{email[0]}'\n"

    llm = OpenAI(temperature=0.6, model='gpt-3.5-turbo-instruct')
    prompt_animal_name = PromptTemplate(
        input_variables=['context', 'emaill'],
        template=""" Você é um assistente de classificação de e-mail
                Os e-mails podem ser classificados em:
                0 - Suporte técnico
                1 - Comercial
                2 - Administrativo
                Responda apenas com o número que repressenta a classificação.
                Classifique o e-email abaixo:
                {context}
                Baseado nesses e-mails semelhantes e suas respectivas classificações:
                {emaill}
            """
        )
    animal_name_chain = LLMChain(llm=llm, prompt=prompt_animal_name)

    response = animal_name_chain({'context': context, 'emaill': emaill})
    return response['text']

# Exemplo de uso
query_email = "Boa tarde, não estou conseguindo baixar o realtório de vendas do sistema"
query_doc = nlp(query_email)
query_vector = np.mean([word.vector for word in query_doc if word.has_vector], axis=0).astype('float32')

similar_emails_indices = search_similar_email(query_vector)
similar_emails = [emails[i] for i in similar_emails_indices]

context = '\n'.join([email[0] for email in similar_emails])

classification = classify_email(similar_emails, context)
print("Classificação do e-mail:", classification)