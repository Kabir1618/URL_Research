import os
import streamlit as st
import pickle
import pinecone
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("Research Tool 📈")
st.sidebar.title("Article URLs")


urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

main_placeholder = st.empty()


query = main_placeholder.text_input("Question: ")
if query:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))

    pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment="gcp-starter"
    )
    index_name = "langchainvector"
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    def retrieve_query(mquery, k=3):
        matching_results = index.similarity_search(mquery, k=k)
        return matching_results
    llm = OpenAI(temperature=0.5)
    chain = load_qa_chain(llm, chain_type="stuff")
    def retrieve_ans(mquery):
        doc_search = retrieve_query(mquery)
        print(doc_search)
        response = chain.run(input_documents = doc_search, question=query)
        return response   
    

    result = retrieve_ans(query)
    st.header("Answer")
    st.write(result)
    # Display sources, if available
    # sources = result.get("sources", "")
    # if sources:
    #     st.subheader("Sources:")
    #     sources_list = sources.split("\n")  # Split the sources by newline
    #     for source in sources_list:
    #     st.write(source)



