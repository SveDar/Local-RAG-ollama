import gradio as gr
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama

# Function to load, split, and retrieve documents
def load_and_retrieve_docs(url):
    """
    Load documents from a given URL, split them into chunks, and create a vector store.

    Args:
        url (str): The URL of the webpage to load the documents from.

    Returns:
        Retriever: A retriever object that can be used to retrieve documents based on a query.
    """
    # Load the documents from the webpage
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict()  # Pass any additional arguments to BeautifulSoup
    )
    docs = loader.load()

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create a vector store using Ollama embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Return the vector store as a retriever
    return vectorstore.as_retriever()

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function that defines the RAG chain
def rag_chain(url, question):
    """
    Function that defines the RAG chain.

    Args:
        url (str): The URL of the webpage to load the documents from.
        question (str): The question to be answered.

    Returns:
        str: The response from the RAG chain.
    """
    # Load documents from the webpage, split them into chunks, and create a vector store
    retriever = load_and_retrieve_docs(url)

    # Retrieve relevant documents based on the question
    retrieved_docs = retriever.invoke(question)

    # Format the retrieved documents into a string
    formatted_context = format_docs(retrieved_docs)

    # Format the question and context into a prompt for the Ollama model
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"

    # Call the Ollama model to generate the response
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])

    # Return the response
    return response['message']['content']

# Gradio interface prebuild forms components and logic how to function
iface = gr.Interface(
    fn=rag_chain,
    inputs=["text", "text"],
    outputs="text",
    title="RAG Chain Question Answering",
    description="Enter a URL and a query to get answers from the RAG chain."
)

# Launch the app
iface.launch()