from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain_community.document_loaders.csv_loader import CSVLoader



# PDF Doc
# loader = PyPDFLoader("Tan Summer Contract  2024 (1).pdf")
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
# texts = text_splitter.split_documents(docs)


# CSV Files
loader = CSVLoader(file_path='30-movies-only-text.csv', encoding='latin1')
texts = loader.load()


# embeding
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# get current device
device = 'cpu'

# init embed model
embed = HuggingFaceEmbeddings(model_name=embed_model_name,\
                                       model_kwargs={"device":device},\
                                       encode_kwargs={"device":device,\
                                       "batch_size":200})



CONNECTION_STRING = "postgresql+psycopg2://postgres:Whatever123:)@localhost:5432/movies"
COLLECTION_NAME = '30_movies_vector'


# Create a vector database
db = PGVector.from_documents(
    embedding=embed,
    documents=texts,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    
)









    # docker run --name pgvector-demo -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d ankane/pgvector