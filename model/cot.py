
import openai
import os
import sys
from dotenv import load_dotenv
load_dotenv()
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

# API KEY from OPENAI website
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# This function is used to pass the argument with query.
query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

# Load the custom Dataset and split into chunks
loader = DirectoryLoader("mydata/")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Embedding the docs and store it into Vector DB and initialize the retriever
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

# Create a customized prompt for Chain of Thought reasoning
def chain_of_thought_prompt(query, chat_history):
    prompt = "Think step-by-step to solve the following question:\n\n"
    for i, (q, a) in enumerate(chat_history):
        prompt += f"Q{i+1}: {q}\nA{i+1}: {a}\n"
    prompt += f"Q{len(chat_history)+1}: {query}\nA{len(chat_history)+1}: Let's think step by step.\n"
    return prompt

# Create the ConversationalRetrievalChain with a custom prompt for Chain of Thought
def chain_of_thought_chain(retriever):
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=retriever,
        return_source_documents=True
    )

chain = chain_of_thought_chain(docsearch.as_retriever(search_kwargs={"k": 1}))

# Initialize an empty list called chat_history to store the conversation history.
chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    
    # Construct the chain of thought prompt
    prompt = chain_of_thought_prompt(query, chat_history)
    result = chain({"question": query, "chat_history": chat_history, "prompt": prompt})
    
    print(result['answer'])

    chat_history.append((query, result['answer']))
    query = None
