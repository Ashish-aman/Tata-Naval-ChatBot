# import os
# import streamlit as st
# from dotenv import load_dotenv
# # from llama_index.prompts import PromptHelper
# import streamlit as st
# # from llama_index import VectorStoreIndex, ServiceContext, Document
# # from llama_index.llms import OpenAI
# import openai
# from llama_index.core import SimpleDirectoryReader
# from llama_index.core import GPTVectorStoreIndex, VectorStoreIndex, ServiceContext, Document
# # from llama_index import  PromptHelper, ServiceContext, download_loader, SimpleDirectoryReader
# from langchain import OpenAI
# from llama_index.llms.openai import OpenAI
# from llama_index.core import SimpleDirectoryReader
# from langchain.callbacks import get_openai_callback
# from PyPDF2 import PdfReader


# # Load environment variables for the OpenAI API key
# load_dotenv()
# # @st.cache_resource(show_spinner=False)
# # def load_data():
# #     with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
# #         reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
# #         docs = reader.load_data()
# #         service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."))
# #         index = VectorStoreIndex.from_documents(docs, service_context=service_context)
# #         return index

# # index = load_data()
# # Set the OpenAI API key
# openai_api_key = os.getenv('OPENAI_API_KEY')
# if not openai_api_key:
#     st.error("OpenAI API key is missing. Please set it in the .env file.")
#     st.stop()

# # Streamlit App UI
# def main():
#     # Streamlit app UI
#     # st.title("Tata Naval: Chat Bot and Troubleshooter for Technical Help Desk")
    
#     # # Instruction text
#     # st.write("Upload a PDF manual, ask your question, and get help from the chatbot!")
    
#     # Step 1: Upload PDF
#     uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")
#     import streamlit as st
#     from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
#     from llama_index.llms import OpenAI
    
#     @st.cache_resource(show_spinner=False)
#     def load_data(uploaded_files):
#         with st.spinner(text="Loading and indexing the uploaded documents – hang tight! This may take a moment."):
#             # Create a list to hold the uploaded documents
#             docs = []
            
#             # Read the uploaded files
#             for uploaded_file in uploaded_files:
#                 # Assuming the uploaded file is a text file, you can adjust this based on your file types
#                 content = uploaded_file.read()
#                 # You might need to decode the content based on the file type
#                 docs.append(content.decode("utf-8"))  # Change this according to your requirements
                
#             # Create a SimpleDirectoryReader using the list of documents
#             # (This is a simplification; you might want to store them in a temporary directory)
#             service_context = ServiceContext.from_defaults(
#                 llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features.")
#             )
            
#             # Indexing the documents
#             index = VectorStoreIndex.from_documents(docs, service_context=service_context)
#             return index
#     chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, streaming = True)
#     if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})

#     for message in st.session_state.messages: # Display the prior chat messages
#         with st.chat_message(message["role"]):
#             st.write(message["content"])

#         # If last message is not from assistant, generate a new response
#     if st.session_state.messages[-1]["role"] != "assistant":
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 response = chat_engine.chat(prompt)
#                 st.write(response.response)
#                 message = {"role": "assistant", "content": response.response}
#                 st.session_state.messages.append(message) # Add response to message history
#     # Streamlit app interface
#     st.title("Document Upload and Indexing")
    
#     # File uploader widget
#     uploaded_files = st.file_uploader("Upload your documents", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    
#     if uploaded_files:
#         # Call load_data only if there are uploaded files
#         index = load_data(uploaded_files)
#         st.success("Documents have been successfully loaded and indexed!")

#     if uploaded_pdf:
#         # Step 2: Read the uploaded PDF
#         reader = PdfReader(uploaded_pdf)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text()
#         with open("temp_manual.txt", "w") as f:
#             f.write(text)
        
#         # Use the LlamaIndex library to load the document and create an index
#         documents = [text]  
    
#         # Define the LLM (OpenAI GPT) model
#         llm_predictor = OpenAI(model_name='gpt-3.5-turbo-instruct')
    
#         # Define prompt helper settings
#         max_input_size = 4096
#         num_output = 512
#         max_chunk_overlap = 20
#         # prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    
#         # Set up the service context with the prompt helper and LLM predictor
#         service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    
#         # # Load documents from the 'data' folder
#         # SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
#         # loader = SimpleDirectoryReader('./data', recursive=True, exclude_hidden=True)
#         # documents = loader.load_data()
    
#         # Create the GPT VectorStore Index using the loaded documents
#         index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
#         query_engine = index.as_query_engine()
    
#     # Set up the chatbot index
#     index = setup_chatbot()
    
#     # Define the query function
#     def get_chat_response(query):
#         query_engine = index.as_query_engine()
#         with get_openai_callback() as cb:
#             response = query_engine.query(query)
#             return response.response.replace('\n\n', '\n'), cb
#             chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)



#     st.set_page_config(page_title="Tata Naval Chatbot & Troubleshooter", layout="wide")

#     # Header
#     st.title("Tata Motors Naval: Technical Helpdesk Chatbot")

#     st.write("Welcome to the Tata Motors Naval Chatbot. You can ask any technical queries related to the vehicles or manuals, and the chatbot will provide troubleshooting help and answers.")

#     # Sidebar Information
#     st.sidebar.title("About the App")
#     st.sidebar.info("This app is designed to help Tata Motors customers and technicians troubleshoot vehicle issues by providing direct answers to questions based on car manuals and other technical documents.")

#     # Query Input
#     user_query = st.text_input("Enter your query below:", placeholder="e.g., How can I open the tailgate?")

#     if st.button("Get Answer"):
#         if user_query:
#             with st.spinner("Fetching the best answer for you..."):
#                 # Get the chatbot response
#                 response, callback_info = get_chat_response(user_query)

#             # Display the result
#             st.subheader("Chatbot Response:")
#             st.write(response)

#             # Optionally, display callback info (OpenAI token usage details)
#             with st.expander("See details of API call (token usage)"):
#                 st.write(callback_info)
#         else:
#             st.error("Please enter a query to get an answer.")

# #     # Footer
# #     st.write("**Note**: This chatbot provides information based on the data available in the technical documents like car manuals. For more detailed technical assistance, contact the Tata Motors Technical Support Team.")

# if __name__ == "__main__":
#     main()




# import os
# import streamlit as st
# from dotenv import load_dotenv
# from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex, ServiceContext, Document
# from langchain import OpenAI
# from langchain.callbacks import get_openai_callback
# from PyPDF2 import PdfReader

# # Load environment variables for the OpenAI API key
# load_dotenv()

# # Set the OpenAI API key
# openai_api_key = os.getenv('OPENAI_API_KEY')
# if not openai_api_key:
#     st.error("OpenAI API key is missing. Please set it in the .env file.")
#     st.stop()

# # Streamlit App UI
# def main():
#     st.title("Document Upload and Indexing")

#     # File uploader widget
#     uploaded_files = st.file_uploader("Upload your documents", type=["txt", "pdf", "docx"], accept_multiple_files=True)

#     if uploaded_files:
#         index = load_data(uploaded_files)
#         st.success("Documents have been successfully loaded and indexed!")

#     if uploaded_pdf:
#         reader = PdfReader(uploaded_pdf)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text()
#         with open("temp_manual.txt", "w") as f:
#             f.write(text)
        
#         documents = [text]  
#         llm_predictor = OpenAI(model_name='gpt-3.5-turbo-instruct')
#         service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
#         index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
#         query_engine = index.as_query_engine()

#     st.set_page_config(page_title="Tata Naval Chatbot & Troubleshooter", layout="wide")
#     st.title("Tata Motors Naval: Technical Helpdesk Chatbot")
#     st.sidebar.title("About the App")
#     st.sidebar.info("This app is designed to help Tata Motors customers and technicians troubleshoot vehicle issues by providing direct answers to questions based on car manuals and other technical documents.")

#     user_query = st.text_input("Enter your query below:", placeholder="e.g., How can I open the tailgate?")
#     if st.button("Get Answer"):
#         if user_query:
#             with st.spinner("Fetching the best answer for you..."):
#                 response, callback_info = get_chat_response(user_query)
#             st.subheader("Chatbot Response:")
#             st.write(response)

# @st.cache_resource(show_spinner=False)
# def load_data(uploaded_files):
#     with st.spinner(text="Loading and indexing the uploaded documents – hang tight! This may take a moment."):
#         docs = []
#         for uploaded_file in uploaded_files:
#             content = uploaded_file.read()
#             docs.append(content.decode("utf-8"))
#         service_context = ServiceContext.from_defaults(
#             llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features.")
#         )
#         index = VectorStoreIndex.from_documents(docs, service_context=service_context)
#         return index

# def get_chat_response(query):
#     query_engine = index.as_query_engine()
#     with get_openai_callback() as cb:
#         response = query_engine.query(query)
#         return response.response.replace('\n\n', '\n'), cb

# if __name__ == "__main__":
#     main()












# !pip install llama_index

# !pip install python-dotenv

# !pip install --upgrade sqlalchemy

import os
from dotenv import load_dotenv

load_dotenv()

# Now you can access the OpenAI API key using the environment variable:
os.environ.get('OPENAI_API_KEY')

from llama_index.core import SimpleDirectoryReader
from llama_index.core import download_loader

# # Downloads the document to be used in the chatbot
# SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
# loader = SimpleDirectoryReader('/content/data', recursive=True, exclude_hidden=True)
# documents = loader.load_data()
import streamlit as st
import os
import tempfile
from llama_index.core import download_loader

# Download the SimpleDirectoryReader loader
SimpleDirectoryReader = download_loader("SimpleDirectoryReader")

# File uploader
uploaded_file = st.file_uploader("Choose a file or manual from which you want to ask a question?")

if uploaded_file is not None:
    # Create a temporary directory to store the uploaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the file path
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Write the uploaded file to the temporary directory
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Write the file content
        
        # Initialize the SimpleDirectoryReader to load from the temporary directory
        loader = SimpleDirectoryReader(input_dir=temp_dir, recursive=False, exclude_hidden=True)
        documents = loader.load_data()
        
        # You can now process the documents and proceed with further steps, e.g., querying the model
        st.write("File successfully loaded. You can now ask your questions.")


# uploaded_file = st.file_uploader("Choose a file or manual from which you want to ask a question?")
# if uploaded_file is not None:
#     # To read file as bytes:
#     loader = SimpleDirectoryReader('/content/data', recursive=True, exclude_hidden=True)
#     documents = loader.load_data()


# !pip install langchain

# !pip install llama-index

import os
os.environ['OPENAI_API_KEY'] =  'sk-R-ufHOxSH8tEH8RxeDHlMQOjvRdwVrq9s6KkzaXexDT3BlbkFJ2l1X9fb-fSwtVaOdfYJjt69TvgJ5ycDs1zS-0Mc0IA'

# Import necessary classes from the LlamaIndex library
from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex, ServiceContext, Document
from langchain import OpenAI
# Import to OpenAI class from langchain library
from langchain import OpenAI

# Defines the language model for the LLMPredictor
# In this case, it is an OpenAI template with temperature 0 and name "text-davinci-003".
llm_predictor = OpenAI(model_name='gpt-3.5-turbo-instruct')

# sets PromptHelper parameters
# The maximum input size for the prompt is 4096 characters
# The maximum output number is 256 tokens
# The maximum chunk overlap is 20 characters
max_input_size = 4096
num_output = 256
max_chunk_overlap = 20
# prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900

# a vector store index only needs an embed model
index = GPTVectorStoreIndex.from_documents(
    documents, embed_model=Settings.embed_model)

# ... until you create a query engine
query_engine = index.as_query_engine(llm=Settings.llm)

# pip install langchain-community --trusted-host mirrors.cloud.tencent.com

from langchain.callbacks import get_openai_callback

# Creates the query engine
query_engine = index.as_query_engine()

# Make the query using the example query string
with get_openai_callback() as cb:
    query_str = "What is the fault code of DFC for wire diagnosis of downstream NOx sensor?"
    response = query_engine.query(query_str)
    print(cb)

formatted_response = response.response.replace('\n\n', '\n')
print(formatted_response)
