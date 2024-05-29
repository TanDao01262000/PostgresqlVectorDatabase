from langchain_community.llms import Ollama 
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
import time
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


# embeding
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# get current device
device = 'cpu'

# init embed model
embeddings = HuggingFaceEmbeddings(model_name=embed_model_name,\
                                       model_kwargs={"device":device},\
                                       encode_kwargs={"device":device,\
                                       "batch_size":200})



CONNECTION_STRING = "postgresql+psycopg2://postgres:Whatever123:)@localhost:5432/vectorDB_test"
COLLECTION_NAME = 'tan_contract_vector'

pgvector_docsearch = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    use_jsonb=True,
)



# LLM model init
llm = Ollama(model="llama3")
# print(llm.invoke("tell me a random fact about water"))

# RAG prompt
template = '''
You are Elevent, Tan Dao's Personal AI Assistant.
You will do your best to answer any question of regarding to Tan Dao.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

NOTES:
Always answer in very positive way and deep detailed
Always be optimistic and bring positive energy in conversation with the user. 
Always ask users if they need any more help at the end of each answer

Context: {context}
Human: {question}
Assistant:"""
'''
prompt = PromptTemplate(
    template=template,
    input_variables=[
        'context',
        'question',
    ]
)

# Initialize RAG pipelines
rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=pgvector_docsearch.as_retriever(search_kwargs={'k': 7}),
    chain_type_kwargs={"prompt": prompt}
)

print(rag_pipeline.invoke('who is the contractor?'))