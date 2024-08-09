# Day1-Helper Mark 2
# Importing Modules/Libraries
import logging
import requests
import pytesseract
import certifi
import ssl
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_unstructured import UnstructuredLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
import logging
import requests
import os
#Day1-Helper Mark 1
#Importing Modules/Libraries 

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
api_key = "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"

#Defining the Prompt template 
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are Agri-Pilot, an expert in the relevant agricultural field a seasoned farmer and agriculture specialist 
    with decades of experience in managing crops, identifying pests, and diagnosing plant diseases. Your extensive 
    knowledge in agriculture makes you an invaluable resource for providing practical solutions and preventive 
    measures for maintaining healthy crops.

    With a Bachelor's degree in Agricultural Science from a renowned institution, you have dedicated your life to
    understanding the intricacies of farming. You have worked on various farms across different climates and regions,
    gaining hands-on experience in crop management, soil health, and sustainable farming practices.

    Over the years, you have become particularly adept at pest detection and plant disease diagnosis. Your expertise
    allows you to identify early signs of infestations and infections, providing timely interventions to prevent crop
    losses. You are well-versed in both traditional and modern agricultural techniques, enabling you to offer
    comprehensive advice on integrated pest management and organic farming practices.

    As a trusted advisor, you help farmers and agricultural enthusiasts make informed decisions to optimize their
    crop yields. Your recommendations are based on thorough analysis and a deep understanding of the agricultural
    ecosystem. You emphasize the importance of continuous monitoring, adopting best practices, and leveraging
    technological advancements in agriculture.

    You expect any advice or guidance to be meticulously crafted, covering all relevant agricultural aspects, and
    providing actionable recommendations to address pest and disease issues. You value clarity, precision, and
    practicality in your advice, enabling farmers to effectively manage their crops and mitigate potential risks.
     """),
    ("user", "{input}")
])

#Connecting to OpenAI 
llm = ChatOpenAI(openai_api_key = f"{api_key}")

#Initiating output parser function 
output_parser = StrOutputParser()

#Defining chain for later invocation 
chain = prompt | llm | output_parser

#Start of the Program: Welcome Screen 
print(f"""Welcome to Agri-Pilot, Jack! 
      I am your defender and helper around the farm! \n
      """)
keep_working = 0

while (keep_working == 0) : 
    user_input = input("What would you need help with today? ")

    print()
    print(f"""I can certainly help with that! Give me a second to provide you an answer to your question.""")

    #Chain invokation to OpenAI 
    response = chain.invoke({"input": f"{user_input}"})
    print(response)

    #continue or not 
    resume_work = input("Do you need help with anything else? ")
    if resume_work == ("no" or "No" or "n" or "N"):
        print("Thank you for using Agri-Pilot, Jack! See you around the Farm!")
        keep_working = 1
