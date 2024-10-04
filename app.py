import streamlit as st
from utils import generate_output

# Streamlit-based UI
def run_streamlit_app():
    """Runs the Streamlit app."""
    st.title("LLM Model Query Interface")

    # Text input for query
    query = st.text_input("Enter your query:", value="Give python code for making sockets.")

    # Checkbox to toggle the use of RAG
    to_rag = st.checkbox("Use RAG for query augmentation", value=False)

    # Generate button
    if st.button("Generate Answer"):
        with st.spinner('Generating answer...'):
            output = generate_output(query=query, to_rag=to_rag)
            st.text_area("Generated Output", value=output, height=300)


# Run the Streamlit app
if __name__ == "__main__":
    run_streamlit_app()
