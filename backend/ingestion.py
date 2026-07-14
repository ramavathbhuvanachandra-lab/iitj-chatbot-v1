from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from backend.embedding import embeddings

def load_documents(data_path: str):
    """
    Load all .docx documents from the given directory.
    """

    loader = DirectoryLoader(
        data_path,
        glob="*.docx",
        loader_cls=Docx2txtLoader,
    )

    documents = loader.load()

    return documents



def split_documents(documents):
    """
    Split loaded documents into chunks for embedding.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(documents)
    print("=" * 80)

  




    return chunks




def create_vectorstore(chunks):
    """
    Create and persist the Chroma vector database.
    """

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="iitj_v1",
        persist_directory="./chroma_db",
    )

    return vectorstore