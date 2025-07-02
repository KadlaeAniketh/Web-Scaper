from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully:\n\n"
    "1. Only extract data matching the description: {parse_description}.\n"
    "2. Return only the matched data—no extra explanation or comments.\n"
    "3. If nothing matches, return an empty string ''."
)

model = OllamaLLM(model="gemma:2b")

def parse_with_ollama(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    results = []
    for i, chunk in enumerate(dom_chunks, 1):
        response = chain.invoke({
            "dom_content": chunk,
            "parse_description": parse_description
        })
        print(f"Parsed chunk {i}/{len(dom_chunks)}")
        results.append(response)

    return "\n".join(results)
