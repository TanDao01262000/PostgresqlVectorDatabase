from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

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


# load the table
pgvector_docsearch = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embed,
)


def search(docsearch, query):
    docs = docsearch.similarity_search(query, k=1)
    result = [doc.page_content for doc in docs]
    print(docs)
    return result

# def add()


if __name__ == "__main__":
    query ='Roxy'
    print(search(pgvector_docsearch, query))