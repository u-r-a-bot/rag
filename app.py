import streamlit as st
import multiprocessing
import os
from utils import LLMModelHandler
import time

@st.cache_resource
def load_model_handler(model_id):
    with st.spinner("Loading Model..."):
        llm_handler = LLMModelHandler(model_id=model_id)
        llm_handler.load_model()
    return llm_handler


model_choice = st.selectbox(
    "Choose the LLaMA model to use:",
    ( "meta-llama/Llama-3.2-1B-Instruct","meta-llama/Llama-3.2-3B-Instruct")
)
llm_handler = load_model_handler(model_id=model_choice)

st.title("LLAMA :llama: chatbot")
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "faiss_index_built" not in st.session_state:
    st.session_state.faiss_index_built = False
if "TO_AUGMENT" not in st.session_state:
    st.session_state.TO_AUGMENT = False

uploaded_file = st.file_uploader(label='Upload a PDF file to retrieve context from', type=['pdf'])

if uploaded_file is not None and not st.session_state.faiss_index_built:
    file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)

    my_bar = st.progress(0, text="Reading text")
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    my_bar.progress(33, text="Reading Complete, Now processing")

    llm_handler.build_faiss_index(file_path)
    my_bar.progress(70, text="Processing Complete, now loading embeddings")

    llm_handler.load_faiss_index()
    my_bar.progress(100, 'Loading Complete')

    st.session_state.faiss_index_built = True
    st.session_state.TO_AUGMENT = True
    uploaded_file.close()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):

    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):

        response_placeholder = st.empty()
        response_text = "" 
        dialogue_template = st.session_state.messages
        dialogues = []
        for x in dialogue_template:
            dialogues.append(x)
        

        for token in llm_handler.generate_output_streaming(prompt, to_rag=st.session_state.TO_AUGMENT, context=dialogues):
            # for char in token:  
            #     response_text += char
            #     response_placeholder.markdown(response_text)  
            #     time.sleep(0.03)
            response_text += token
            response_placeholder.markdown(response_text)  
            


    st.session_state.messages.append({"role": "assistant", "content": response_text})
