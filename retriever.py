from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_myforge_retriever():
    loader = DirectoryLoader("knowledge_base", glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    splits = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings(), persist_directory="./chroma_db")
    return vectorstore.as_retriever(search_kwargs={"k": 5})