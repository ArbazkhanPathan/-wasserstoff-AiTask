import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Initialize Flask app
app = Flask(__name__)

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

@app.route('/query', methods=['POST'])
def query_api():
    data = request.json
    query = data.get('query', None)

    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    # Construct the chain of thought prompt
    prompt = chain_of_thought_prompt(query, chat_history)
    result = chain({"question": query, "chat_history": chat_history, "prompt": prompt})
    
    answer = result['answer']
    chat_history.append((query, answer))

    return jsonify({'answer': answer})

@app.route('/new_post', methods=['POST'])
def fetch_new_wordpress_posts():
    data = request.json
    base_url = data.get('base_url', None)

    if not base_url:
        return jsonify({'error': 'Base URL parameter is required'}), 400
    
    # Define the URL of the WordPress site's REST API
    posts_endpoint = f'{base_url}/posts'

    # Make a GET request to fetch posts
    response = requests.get(posts_endpoint)

    if response.status_code == 200:
        # Extract only the content from each post
        posts = response.json()
        
        # Extract the site name from the base_url for filename
        site_name = base_url.split('//')[-1].split('.')[0]
        # filename = f"{site_name}_posts_content.txt"
        filename = f"mydata/{site_name}_posts_content.txt"

        # Write the extracted content to a text file
        with open(filename, 'w', encoding='utf-8') as txt_file:
            for post in posts:
                content = post['content']['rendered']
                # Use BeautifulSoup to remove HTML tags
                soup = BeautifulSoup(content, 'html.parser')
                text_content = soup.get_text()
                txt_file.write(text_content + "\n" + "="*80 + "\n")  # Separator between posts
            
        return f"Posts content fetched and stored in '{filename}'", 200
    else:
        return f"Failed to fetch posts: {response.status_code}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
