from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv


load_dotenv()

#Create your own huggingface account and create your API KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get('HUGGINGFACE_KEY')

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=1000,
    do_sample=False,
)
print(llm.invoke("write python code for binary sort"))