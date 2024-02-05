import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from dotenv import load_dotenv

load_dotenv()

# Vetorizar os dados contidos no arquivo CSV

loader = CSVLoader(file_path="pibmunicipios.csv", encoding="utf8")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)


# Função para fazer a Busca por Similaridade

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    print(page_contents_array)

    return page_contents_array



# Configurar LLM e Prompts

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
Assistente é uma IA de consulta de dados que tira dúvidas a respeito dos dados fornecidos no formato CSV.
Assistente interpreta os dados CSV em formato de tabela para a melhor compreensão do usuário.
Assistente elabora respostas precisas com base no contexto fornecido.
Assistente fornece referências extraídas do contexto abaixo. Não gere links ou referências adicionais. Não consulte conteúdos externos aos dados fornecidos.
Ao final da resposta, exiba no formato de lista as referências extraídas.
Caso não consiga encontrar informações suficientes para a resposta no contexto abaixo, diga apenas 'Infelizmente, não tenho informações suficientes para responder a esta pergunta :('
Caso a pergunta não esteja dentro do contexto dos dados, diga apenas 'Infelizmente, não tenho informações suficientes para responder a esta pergunta :('

Pergunta: {query}
Contexto: {context}
"""

prompt = PromptTemplate(
    input_variables=["query", "context"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# 4. Retrieval augmented generation
def generate_response(query):
    context = retrieve_info(query)
    response = chain.run(query=query, context=context)
    return response


query = "Qual o municipio com o maior PIB da lista?"
response = generate_response(query)

print(response)
