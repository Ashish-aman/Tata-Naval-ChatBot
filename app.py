import os
import streamlit as st
import tempfile
import sqlite3
from datetime import datetime
from llama_index.core import download_loader
from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex, ServiceContext, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

# Database setup (SQLite in this example)
DB_FILE = "chat_data.db"

def init_db():
    """Initialize the SQLite database and create the chat_data table."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS chat_data
                   (timestamp TEXT, customer_name TEXT, car_model TEXT, warranty_date TEXT, query TEXT, response TEXT)''')
    conn.commit()
    conn.close()

def save_chat_data(customer_name, car_model, warranty_date, query, response):
    """Save the chat data (customer info, query, and response) to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("INSERT INTO chat_data VALUES (?, ?, ?, ?, ?, ?)", 
                (timestamp, customer_name, car_model, str(warranty_date), query, response))
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Streamlit app title and description
st.title("Tata Naval ChatBot: Troubleshooter for your Car")
st.subheader("Query your car issues, ask for faults, or check information by uploading relevant documents.")

# Customer information input
st.write("### Enter Customer Information")
customer_name = st.text_input("Customer Name")
car_model = st.text_input("Car Model")
warranty_date = st.date_input("Warranty Expiry Date")

# Upload documents
st.write("### Upload Documents")
uploaded_file = st.file_uploader("Upload the manual or relevant documents", type=["pdf", "txt", "docx"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Initialize the document reader to load from the temporary directory
        loader = SimpleDirectoryReader(input_dir=temp_dir, recursive=False, exclude_hidden=True)
        documents = loader.load_data()

        # Model settings
        Settings.llm = OpenAI(model="gpt-3.5-turbo")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        Settings.num_output = 512
        Settings.context_window = 3900

        # Build vector store index
        index = GPTVectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)

        # Create query engine
        query_engine = index.as_query_engine(llm=Settings.llm)

        st.success("Documents uploaded and indexed successfully!")

        # Display dashboard for asking queries
        st.write("### Ask a Question")
        query_str = st.text_input("Enter your question here")

        # Button to shoot the query
        if st.button("Shoot"):
            if query_str:
                with get_openai_callback() as cb:
                    response = query_engine.query(query_str)
                
                # Format and display the response
                formatted_response = response.response.replace('\n\n', '\n')
                st.write(f"**Response:**\n\n{formatted_response}")

                # Save the chat data to SQLite
                save_chat_data(customer_name, car_model, warranty_date, query_str, formatted_response)
                st.success("Chat data saved successfully!")
            else:
                st.error("Please enter a question before shooting.")

# Option to view all stored chat data (Optional)
if st.button("View Chat Data"):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM chat_data", conn)
    st.dataframe(df)
    conn.close()
