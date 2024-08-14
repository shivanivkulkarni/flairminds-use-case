import streamlit as st
from uuid import uuid4
from parent_child_retriever import *

directory = f"Flairminds Docs"

# loading all csv and pdf files from the specified property directory
load_files_and_vectorstore(directory)





