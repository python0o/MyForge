import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_myforge_retriever():
    kb_path = "knowledge_base"

    if not os.path.exists(kb_path):
        print("Warning: knowledge_base folder not found. Returning empty retriever.")
        return None

    loader = DirectoryLoader(kb_path, glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()

    if not docs:
        print("Warning: No documents found in knowledge_base.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})
