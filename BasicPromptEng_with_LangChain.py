import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate 
from langchain.chains.llm import LLMChain


load_dotenv()
GOOGLE_API_TOKEN = os.environ.get('GOOGLE_API_KEY')

llm = GoogleGenerativeAI(model="models/gemini-pro", google_api_key=GOOGLE_API_TOKEN)

st.title ('Basic Prompt Engineering')
financial_concept_input = st.text_input("Search the topic in financial concept")
temp_template = '''I want you to act as financial expert. For a common man, explain the basics of  {financial_concept}'''

finance_prompt = PromptTemplate(input_variables=['financial_concept'],
                                    template= temp_template)

finance_prompt.format(financial_concept=financial_concept_input)

finance_chain = LLMChain(llm=llm, prompt=finance_prompt)

st.write(finance_chain.run(financial_concept_input))

translate_sentence = st.text_input('Enter a sentence to translate')
target_language = st.selectbox("Select a language to translate:",options=['Tamil','Telugu', 'Tamil','Malay','Mandarin'],index=0)

st.write(target_language)

template='''Translate the following sentence 'Welcome' into Tamil'''
language_prompt = PromptTemplate(
    input_variables=["sentence",'target_language'],
    template=template,
)
#language_prompt.format(sentence=translate_sentence,target_language=target_language)

st.write(language_prompt)
language_chain = LLMChain(llm=llm, prompt=language_prompt)
if (translate_sentence):
    result = language_chain({'sentence':'Welcome', 'target_language':'Tamil'})
    st.write(result)