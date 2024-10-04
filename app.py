import streamlit as st
import os
from utils import LLMModelHandler
import time

@st.cache_resource
def load_model_handler():
    with st.spinner("Loading Model..."):
        llm_handler = LLMModelHandler()
        llm_handler.load_model()
    return llm_handler

llm_handler = load_model_handler()

st.title("Echo Bot")
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)
TO_AUGMENT = False
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "faiss_index_built" not in st.session_state:
    st.session_state.faiss_index_built = False


uploaded_file = st.file_uploader(label='Upload a PDF file to retrieve context from', type=['pdf'])
if uploaded_file is not None and not st.session_state.faiss_index_built:
    file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
    my_bar = st.progress(0, text="Reading text")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    my_bar.progress(33 , text = "Reading Complete, Now processing")
    llm_handler.build_faiss_index(file_path)
    my_bar.progress(70 , text = "Processing Complete, now loading embeddings ")
    llm_handler.load_faiss_index()
    my_bar.progress(100 , 'Loading Complete')
    TO_AUGMENT = True
     
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])




if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # Create a placeholder for streaming response
        response_placeholder = st.empty()
        response_text = ""  # Variable to accumulate the response
        dialogue_template = st.session_state.messages
        dialogues = []
        for x in dialogue_template:
            dialogues.append(x)
        # Generate tokens one by one
        for token in llm_handler.generate_output_streaming(prompt, to_rag=TO_AUGMENT,context=dialogues):
            # response_text += token  # Accumulate the response text
            # response_placeholder.markdown(response_text)  
            for char in token:  # Iterate over each character in the token
                response_text += char
                response_placeholder.markdown(response_text)  # Update the placeholder with new text
                time.sleep(0.06)
    # Display assistant response in chat message container
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "Assistant", "content": response_text})