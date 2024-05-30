import os
import streamlit as st
from dotenv import load_dotenv
#from langchain.llms import google_palm
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains.sequential import SequentialChain
from langchain_google_genai import GoogleGenerativeAI



load_dotenv()
os.environ["GOOGLE_API_TOKEN"] = os.environ.get('GOOGLE_API_KEY')

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.environ["GOOGLE_API_TOKEN"])

#print(llm.invoke("What are some of the pros and cons of Python as programming language?"))

st.title ('Celebrity Search Results')
input_text = st.text_input("Search the topic u want")

first_input_prompt = PromptTemplate(input_variables=['name'],
                                    template="tell me about celebrity {name}")

person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory =  ConversationBufferMemory(input_key='dob',memory_key='description_history')

chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person', memory=person_memory)

second_input_prompt = PromptTemplate(input_variables=['person'],
                                    template= "when was {person} born")

chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob', memory=dob_memory)

third_input_prompt= PromptTemplate(input_variables=['dob'],
                                   template="Mention 5 major events happend around {dob} in the world")

chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='description', memory=descr_memory)

parent_chain = SequentialChain(chains=[chain,chain2, chain3], input_variables=['name'], 
                               output_variables=['person','dob','description'], verbose=True)

if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(descr_memory.buffer)