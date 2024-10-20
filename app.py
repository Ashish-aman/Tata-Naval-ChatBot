import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext, download_loader, SimpleDirectoryReader
from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from PyPDF2 import PdfReader


# Load environment variables for the OpenAI API key
load_dotenv()

# Set the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    st.error("OpenAI API key is missing. Please set it in the .env file.")
    st.stop()

# Streamlit App UI
def main():
    # Streamlit app UI
    # st.title("Tata Naval: Chat Bot and Troubleshooter for Technical Help Desk")
    
    # # Instruction text
    # st.write("Upload a PDF manual, ask your question, and get help from the chatbot!")
    
    # Step 1: Upload PDF
    uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_pdf:
        # Step 2: Read the uploaded PDF
        reader = PdfReader(uploaded_pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        with open("temp_manual.txt", "w") as f:
            f.write(text)
        
        # Use the LlamaIndex library to load the document and create an index
        documents = [text]  
    
        # Define the LLM (OpenAI GPT) model
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))
    
        # Define prompt helper settings
        max_input_size = 4096
        num_output = 512
        max_chunk_overlap = 20
        prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    
        # Set up the service context with the prompt helper and LLM predictor
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    
        # # Load documents from the 'data' folder
        # SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
        # loader = SimpleDirectoryReader('./data', recursive=True, exclude_hidden=True)
        # documents = loader.load_data()
    
        # Create the GPT VectorStore Index using the loaded documents
        index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
        query_engine = index.as_query_engine()
    
    # Set up the chatbot index
    index = setup_chatbot()
    
    # Define the query function
    def get_chat_response(query):
        query_engine = index.as_query_engine()
        with get_openai_callback() as cb:
            response = query_engine.query(query)
            return response.response.replace('\n\n', '\n'), cb


    st.set_page_config(page_title="Tata Naval Chatbot & Troubleshooter", layout="wide")

    # Header
    st.title("Tata Motors Naval: Technical Helpdesk Chatbot")

    st.write("Welcome to the Tata Motors Naval Chatbot. You can ask any technical queries related to the vehicles or manuals, and the chatbot will provide troubleshooting help and answers.")

    # Sidebar Information
    st.sidebar.title("About the App")
    st.sidebar.info("This app is designed to help Tata Motors customers and technicians troubleshoot vehicle issues by providing direct answers to questions based on car manuals and other technical documents.")

    # Query Input
    user_query = st.text_input("Enter your query below:", placeholder="e.g., How can I open the tailgate?")

    if st.button("Get Answer"):
        if user_query:
            with st.spinner("Fetching the best answer for you..."):
                # Get the chatbot response
                response, callback_info = get_chat_response(user_query)

            # Display the result
            st.subheader("Chatbot Response:")
            st.write(response)

            # Optionally, display callback info (OpenAI token usage details)
            with st.expander("See details of API call (token usage)"):
                st.write(callback_info)
        else:
            st.error("Please enter a query to get an answer.")

    # Footer
    st.write("**Note**: This chatbot provides information based on the data available in the technical documents like car manuals. For more detailed technical assistance, contact the Tata Motors Technical Support Team.")

if __name__ == "__main__":
    main()
