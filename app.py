import glob
from tempfile import NamedTemporaryFile
import ollama
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st


def response_generator(stream):
    for chunk in stream.response_gen:
        yield chunk
        
@st.cache_resource(show_spinner=False)
def load_data(model_name:str) -> VectorStoreIndex:
    llm = Ollama(model=model_name, request_timeout=30.0)
    docs = SimpleDirectoryReader("docs_vectorstore").load_data()

    text_splitter = SentenceSplitter(chunk_size=512)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5") # BAAI/bge-small-en-v1.5 | BAAI/bge-base-en-v1.5
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        text_splitter=text_splitter,
        system_prompt="You are an expert and your job is to answer and explain analytical questions. Do not give me an answer if it is not mentioned in the context as a fact.")

    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index

def main() -> None:
     
    st.set_page_config(page_title="ИИ-исследователь", page_icon="🧠", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("ИИ-исследователь 🧠")
    
    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    with st.sidebar:
        if "model" not in st.session_state:
            st.session_state["model"] = ""
        models = [model["name"] for model in [ollama.list()["models"][1], ollama.list()["models"][2], ollama.list()["models"][3]]]
        st.session_state["model"] = st.selectbox("Выберите модель", models)
        
        llm = Ollama(model=st.session_state["model"], request_timeout=30.0)

        #document = st.file_uploader("Загрузите файл в формате pdf", type=['pdf'], accept_multiple_files=False)
        
        #if st.button('Обработать файл'):
        
        #index = load_data(st.session_state["model"])
        st.session_state.activate_chat = True
        
        index_ret = load_data(st.session_state["model"])
        #query_engine = index_ret.as_query_engine(similarity_top_k=3, streaming=True)
    
    if st.session_state.activate_chat == True:          
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
            index_ret = load_data(st.session_state["model"])
            #st.session_state.chat_engine = index.as_chat_engine(chat_mode="context", streaming=True)
            query_engine = index_ret.as_query_engine(similarity_top_k=3, streaming=True)
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Чем я могу быть полезен?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            if st.session_state.messages[-1]["role"] != "assistant":
                message_placeholder = st.empty()
                with st.chat_message("assistant"):
                    #stream = st.session_state.chat_engine.stream_chat(prompt)
                    res = query_engine.query(str(prompt))
                    #if 'Do not give answer' not in response_generator(res):
                    inds = len(res.source_nodes)
                    r = 'Ответ сгенирирован на основании ' + str(len(res.source_nodes)) +' документов: \n'
                    if inds != 1:
                        for i, node in zip(range(0, inds), res.source_nodes):
                            r += str(i+1) + ") " + node.node.metadata['file_name'].replace('\n', '') + '\n'
                            if 'page_label' in node.node.metadata.keys():
                                r += "Страница номер:" + str(node.node.metadata['page_label']).replace('\n', '') + '\n'
                            r += node.node.text.replace('\n', '') + '\n'
                            r += "Семантическое сходство с запросом:" + str(round(node.score,3)).replace('\n', '')  + '\n'
                            #response = st.write_stream(response_generator(stream))
                    #st.write_stream(response_generator(res))
                    if "expert and my job" in str(res):
                        st.write("According to the provided context information, I will not be able to provide an answer to this question.")
                    else:
                        st.write(str(res))
                    #if 'I will not be able to answer' not in response_generator(res):
                    if 'app.py' not in r:
                        st.write(r)
                    else:
                        r = ''
                    #else:
                    #    st.write('This information was not provided in context and I will not be able to answer that question')
                    st.session_state.messages.append({"role": "assistant", "content": res})
                
    else:
        st.markdown("<span style='font-size:15px;'><b>Для начала разговора загрузите файлы в формате pdf.</span>", unsafe_allow_html=True)

if __name__=='__main__':
    main()
    
#pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116