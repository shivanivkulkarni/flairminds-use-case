import streamlit as st
from parent_child_retriever import *
from uuid import uuid4
from chat_processing import *

st.set_page_config(page_title='Flairminds', layout='wide')

@st.cache_resource
def load_retrival():
    retriever = load_faiss_retriever()
    chain, prompt = get_conversational_chain()
    return retriever, chain, prompt

retriever, chain, prompt = load_retrival()
# Custom CSS for card styling
st.markdown("""
    <style>
    .card {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        padding: 20px;
        text-align: center;
        margin: 10px;
    }
    .card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .card img {
        max-width: 200px;
        margin-bottom: 10px;
    }
    .card h3 {
        margin: 10px 0;
    }
    .card p {
        color: #555;
    }
    .separator {
        border: 0;
        height: 1px;
        background: #ddd;
        margin: 40px 0;
    }
    div.stButton > button:first-child {
        background-color: cornflowerblue; /* Set background color to Cornflower Blue */
        color: white; 
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        border-radius: 50px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: fixed; /* Fixes the position of the button */
        bottom: 20px; /* Adjusts the distance from the bottom */
        right: 20px; /* Adjusts the distance from the right */
        z-index: 1000; /* Ensures the button is on top of other content */
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.1);
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
        cursor: pointer;
    }
    div.stButton > button:first-child:default {
        color: white; /* Ensure text color remains white when button is clicked */
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation or additional content
with st.sidebar:
    # Create a horizontal layout for the logo and title
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image('images/logo-removebg.png', width=60)  # Adjust width as needed
    with col2:
        st.title("Flairminds ")

    st.write("About Us")
    st.write("Contact Us")
    st.write("Team")
    st.write("Careers")

# Main Image at the top
st.image('images/landing_page.png', use_column_width=True)

# About Us Section
st.markdown("<h2 style='text-align: center;'>About Us</h2>", unsafe_allow_html=True)
st.write("Innovation, Efficiency and Resiliency are the keystones of IT industry today and will be so in the future too. At FlairMinds, we are determined to empower businesses with cutting edge innovative technology solutions and streamline software engineering to deliver greater efficiency and enable resiliency across your products and services.")

# Horizontal Line after About Us
st.markdown("<hr class='separator'>", unsafe_allow_html=True)

# Services section
st.markdown("<h2 style='text-align: center;'>Services</h2>", unsafe_allow_html=True)
st.write("We have been recognized by our customers for having executed numerous Modernization projects. Our team of experts in Cloud, DevOps, Agile Software development and Data Science will accompany you during every stage of your cloud journey combining consulting, development, testing and migration. We are looking forward to our collaborative journey to the Digital Future.")

# First row with 3 services as cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="card">
            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQCP7WwqRZMt45muqpTdBtWssi3TJHjGG4UkwvxBaIBRAV5naPe4-XBwl6tatEPaZWkDDk&usqp=CAU" alt="Full Stack"/>
            <h3>Full Stack</h3>
            <p>Modern businesses demand agile, intelligent business and IT applications. In practice, this translates to growing urgency for organizations to move from legacy technologies to cloud-based solutions. Let FlairMinds help you make the leap to the cloud.</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="card">
            <img src="https://www.gyansetu.in/wp-content/uploads/2024/02/FUqHEVVUsAAbZB0-1024x580-1.jpg" alt="Data Science"/>
            <h3>Data Science</h3>
            <p>FlairMinds builds data strategies to get all of your data into a consistent system and ensure end-to-end governance, visibility, and quality. Let us help you unravel layers of legacy systems and solve complex situations.</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="card">
            <img src="https://media.istockphoto.com/id/1303169188/photo/devops-concept.webp?b=1&s=170667a&w=0&k=20&c=6JWGUKuIzdwoHXYCiZsE1GT9IJCjXm5O2LVKsvJcApA=" alt="DevOps"/>
            <h3>DevOps</h3>
            <p>FlairMinds DevOps practice combines software development (Dev) and IT operations (Ops). We aim to shorten and modernize the system's development life cycle providing continuous delivery with high software quality.</p>
        </div>
    """, unsafe_allow_html=True)


@st.dialog("Welcome to FM Chatbot",width="large")
def chat():
    chat_process(retriever, chain, prompt)
    
if "message" not in st.session_state:
    button = st.button("Chat with Flairminds")
    if button:
        chat()
    



