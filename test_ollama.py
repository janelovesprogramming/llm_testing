
import ollama
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


import torch
print(torch.cuda.is_available())




llm = Ollama(model='llama3:8b', request_timeout=30.0)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") # BAAI/bge-small-en-v1.5 | BAAI/bge-base-en-v1.5
    
docs = SimpleDirectoryReader(input_dir="/mnt/sdb1/home/esurikova/llm_testing/texts").load_data()
text_splitter = SentenceSplitter(chunk_size=512)


service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    text_splitter=text_splitter,
    system_prompt="You are an economic expert and your job is to answer analytical questions. Stick to the facts in your answers.")

index = VectorStoreIndex.from_documents(docs, service_context=service_context)

query_engine = index.as_query_engine(similarity_top_k=3)
res = query_engine.query("What is the Cosmic Adaptationism: Black Cloud(s)?")
print(res.response)
'''
for node in res.source_nodes:
    print(node.node.metadata['file_name'])
    print(node.node.metadata['page_label'])
    print(node.node.text)
    print(node.score)
'''
r = 'Ответ сгенирирован на основании 3 документов: \n'
inds = [1, 2, 3]
for i, node in zip(inds, res.source_nodes):
    r += str(i) + ") " + node.node.metadata['file_name'] + '\n'
    r += "Страница номер:" + str(node.node.metadata['page_label']) + '\n'
    r += node.node.text + '\n'
    r += "Семантическое сходство с запросом:" + str(node.score)  + '\n'
print(r)