from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain, ConfigurableField
from dotenv import load_dotenv
import os


class rag_func:
    def __init__(self):
        load_dotenv()
        os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')
        self.init_model_hyde()
        
    def doc_process(self, url):
        # Initialize a web-based document loader and load the document
        loader = WebBaseLoader(url)
        docs = loader.load()

        # Initialize the NVIDIA Embeddings module
        embeddings = NVIDIAEmbeddings()

        # Initialize a text splitter to divide documents into smaller chunks with specified size and overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

        # Split the loaded documents into chunks
        documents = text_splitter.split_documents(docs)

        # Initialize a FAISS vector store from the document chunks and embeddings
        vector = FAISS.from_documents(documents, embeddings)

        # Convert the vector store into a retriever for searching documents
        self.retriever = vector.as_retriever()

    def init_model_hyde(self):
        # Initialize the ChatNVIDIA model with a specified model version as the model to generate hypothetical documents based on a question
        model_hyde = ChatNVIDIA(model="ai-llama3-8b")

        # Define a template for generating hypothetical answers to questions
        hyde_template = [("system", "You are a helpful AI assistant."), ("user", "Generate a one-paragraph hypothetical answer to the below question:{input}")]


        # Initialize a prompt template from the hypothetical answer template
        hyde_prompt = ChatPromptTemplate.from_messages(hyde_template)

        # Chain the prompt template with the model and a string output parser to form a query transformer
        self.hyde_query_transformer = hyde_prompt | model_hyde | StrOutputParser()

    # Define a chainable function to generate hypothetical documents based on a question!
    @chain
    def hyde_retriever(self, question):
        hypothetical_document = self.hyde_query_transformer.invoke({"input": question})
        return self.retriever.invoke(hypothetical_document)


    # Generate final answer
    def infer_request(self, question, max_tokens=1024, temperature=0.1, top_p=0.1):
        model_qa = ChatNVIDIA(model="ai-llama3-8b", max_tokens=max_tokens, temperature=temperature, top_p=top_p,)
        

        template = [("system", "You are a helpful AI assistant."), ("user", "Answer the question strictly based on the following context: {context} Question: {input}")]


        # Initialize a prompt template from the answer template
        prompt = ChatPromptTemplate.from_messages(template)
        # Chain the prompt template with the model_qa and a string output parser to form a query transformer
        answer_chain = prompt | model_qa | StrOutputParser()
        # Generate context with the hypothetical document retriever as documents
        documents = self.hyde_retriever.invoke(self, question=question)
        # Generate response by invoke the chain
        response = answer_chain.invoke({"input": question, "context": documents})
        return response