from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

def generate_cat_name(animal_type, color):
    llm = OpenAI(temperature=0.6)
    prompt_animal_name = PromptTemplate(
        input_variables=['animal_type', 'color'],
        template="Você tem um {animal_type} novo da cor {color} e gostaria de dar um nome para ele. Me de uma lista de 5 possíveis nomes."
    )
    animal_name_chain = LLMChain(llm=llm, prompt=prompt_animal_name)

    response = animal_name_chain({'animal_type': animal_type, 'color': color})

    return response

if __name__=="__main__":
    print(generate_cat_name('gatinho', 'marrom'))
