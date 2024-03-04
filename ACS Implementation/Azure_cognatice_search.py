import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
import openai
import streamlit as st
import tempfile

# Azure Cognitive Search credentials
AZURE_COGNITIVE_SEARCH_ENDPOINT = "YOUR_AZURE_SEARCH_ENDPOINT"
AZURE_COGNITIVE_SEARCH_KEY = "YOUR_AZURE_SEARCH_KEY"
AZURE_COGNITIVE_SEARCH_INDEX_NAME = "YOUR_AZURE_INDEX_NAME"

# Azure OpenAI Credentials
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ['OPENAI_API_TYPE'] = 'azure'
os.environ['OPENAI_API_BASE'] = "YOUR_OPENAI_AZURE_BASE_URL"
os.environ['OPENAI_API_VERSION'] = "YOUR_OPENAI_API_VERSION"

def main():
    st.title("PDF Question Answering App")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Create a temporary file to store the uploaded PDF
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        document_path = temp_file.name  # Use the temp file path

        loader = PyPDFLoader(document_path)
        documents = loader.load_and_split()

        text_splitter = CharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=25
        )
        docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            deployment="text-embedding-ada-002",
            client="azure"
        )

        template = """
        Use the following pieces of context to answer the question at the end.
        If the question is not related to PDF, say that it is not related to PDF, don't try to make up an answer. Understand table values also. Always say "thanks for asking!" at the end of the answer.
        PDF Context: {context}
        Question: {question}
        Helpful Answer:
        """

        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

        vector_store = AzureSearch(
            azure_search_endpoint=AZURE_COGNITIVE_SEARCH_ENDPOINT,
            azure_search_key=AZURE_COGNITIVE_SEARCH_KEY,
            index_name=AZURE_COGNITIVE_SEARCH_INDEX_NAME,
            embedding_function=embeddings.embed_query
        )
        
        vector_store.add_documents(documents=docs)

        llm = AzureOpenAI(
            temperature=0,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            deployment_name="gpt-35-turbo",
            model_name="gpt-35-turbo"
        )

        retriever = vector_store.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm,
                                         retriever=retriever,
                                         return_source_documents=True,
                                         chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

        question = st.text_input("Ask a question:")
        if question:
            result = qa({"query": question})
            answer = result["result"][:-10]  # Extracting the answer
            st.write(answer)

if __name__ == '__main__':
    main()
