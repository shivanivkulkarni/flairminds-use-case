import streamlit as st
from uuid import uuid4
from parent_child_retriever import *

def chat_process(retriever, chain, prompt):
    if "session_uuid" not in st.session_state:
        st.session_state.session_uuid = str(uuid4())

    qna_tests_csv_path = "qna_tests.csv"
    method = "Parent Child Retriever"
   
    # styling for chatbot UI
    st.markdown("""
    <style>
        .st-emotion-cache-1c7y2kd {
            flex-direction: row-reverse;
            text-align: right;
            background-color: rgba(255, 255, 255);
        }
                
        .st-emotion-cache-janbn0 {
            flex-direction: row-reverse;
            text-align: right;
            background-color: rgb(14 17 23);
        }
    </style>
    """,unsafe_allow_html=True)

    text_input_question = st.chat_input("Ask a Question")

    #### Display Chat history messages 
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]
    
    # display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Initialization
    user_question = None
    button_name = None

    # defining text box for adhoc queries 
    if text_input_question:
        user_question = text_input_question
        button_name = user_question

    st.session_state.current_prompt= user_question

    if user_question:
        st.session_state.current_prompt = user_question
        st.session_state.current_button = button_name

        if "current_button" in st.session_state and "current_prompt" in st.session_state:
            st.session_state.messages.append({"role": "user", "content": st.session_state.current_button})
            with st.chat_message("user"):
                st.write(st.session_state.current_button)

            if st.session_state.messages[-1]["role"] != "assistant":
                query_id = uuid4() # generate unique id for each query  
                with st.chat_message("assistant"):
                    with st.spinner("Analysing..."):
                        # call the RAG Chain
                        response, docs = user_input(st.session_state.current_prompt, retriever, st.session_state.session_uuid, query_id, chain, prompt)
                        st.write(response["output_text"])
                message = {"role": "assistant", "content": response["output_text"], "source documents": response["input_documents"]}
                st.session_state.messages.append(message)
                
                # storing query and response in qna_tests.csv
                store_results(st.session_state.session_uuid, query_id, qna_tests_csv_path, method, response, docs)