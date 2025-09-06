from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Choose from: llama3-70b-8192, llama3-8b-8192, gemma-7b-it, etc.
model = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

template = (
    "You are tasked with extracting specific information from the following text content:\n\n{dom_content}\n\n"
    "Please follow these instructions carefully:\n"
    "1. Only extract data matching the description: {parse_description}.\n"
    "2. Return only the matched data—no extra explanation or comments.\n"
    "3. If nothing matches, return an empty string ''."
)

prompt = ChatPromptTemplate.from_template(template)
chain = LLMChain(llm=model, prompt=prompt)

def parse_with_groq(dom_chunks, parse_description):
    results = []
    for i, chunk in enumerate(dom_chunks, 1):
        response = chain.invoke({
            "dom_content": chunk,
            "parse_description": parse_description
        })
        print(f"Parsed chunk {i}/{len(dom_chunks)}")
        results.append(response["text"])
    return "\n".join(results)

