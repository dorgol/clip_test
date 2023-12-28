from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
import yaml
import os
from dotenv import load_dotenv
import openai
import streamlit as st


load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = os.getenv("OPENAI_API_KEY")


with open('config.yaml') as f:
    config = yaml.safe_load(f)
ENRICHMENT_MODEL_NAME = config['ENRICHMENT_MODEL_NAME']


def get_prompt_enrichment(prompt, template):
    llm = ChatOpenAI(model_name=ENRICHMENT_MODEL_NAME, temperature=0.5, openai_api_key=os.environ['OPENAI_API_KEY'])
    summary_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(template))

    summary_response = summary_chain(prompt)
    return summary_response


def run_enrichment(prompt):
    prompt_template = """
    I have a dataset of images. I'm using clip embeddings for search over the images.
    The user will give you a text input. 
    Write a prompt for Clip search based on the user input. The prompt that you will write should return good results
    from clip search.
    Keep the prompt quite short (up to 77 tokens). 
    The user input is: {user_input}
"""

    enrichment = get_prompt_enrichment(prompt, prompt_template)
    return enrichment
