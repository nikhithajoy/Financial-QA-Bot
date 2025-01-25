import os
from dotenv import load_dotenv
import streamlit as st
import nest_asyncio
import google.generativeai as genai
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse
from pinecone import Pinecone, ServerlessSpec
import tempfile

# Load environment variables
load_dotenv()

# Fetch API keys
pinecone_api_key = os.getenv('pinecone_api_key')
gemini_api_key = os.getenv('gemini_api_key')
llama_key = os.getenv('llama_key')

# Streamlit App Configuration
st.set_page_config(page_title="📊💬 Finance Bot")

# Sidebar - File Upload
with st.sidebar:
    st.title('📊💬 Finance Bot')
    uploaded_file = st.file_uploader("Upload your PDF containing P&L statement", type=["pdf"])

# Initialize session state for messages and embeddings
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you with your financial data today?"}
    ]

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
    st.session_state.index = None

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Clear chat history function
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you with your financial data today?"}
    ]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Initialize Pinecone and Embedding
nest_asyncio.apply()
index_name = 'hybrid-search-langchain-pinecone'
pc = Pinecone(api_key=pinecone_api_key)

# Create Index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='dotproduct',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(index_name)
st.session_state.index = index

# Configure Google Gemini API
genai.configure(api_key=gemini_api_key)

# Function to process PDF and create embeddings
def process_pdf(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    # Configure LlamaParse
    os.environ["LLAMA_CLOUD_API_KEY"] = llama_key
    llama_parser = LlamaParse(result_type="markdown")

    # Load PDF document
    documents = llama_parser.load_data(temp_file_path)

    # Initialize embedding model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Text Splitter configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    # Prepare documents for embedding
    docs = []
    for doc in documents:
        texts = text_splitter.split_text(doc.text)
        for text in texts:
            docs.append({
                'page_content': text,
                'metadata': {'source': getattr(doc, 'source', 'Unknown')}
            })

    # Embed documents
    def embed_documents(docs):
        embedded_docs = []
        for doc in docs:
            embedding = embeddings.embed_query(doc['page_content'])
            embedded_docs.append({
                'id': f"doc_{hash(doc['page_content'])}",
                'values': embedding,
                'metadata': {
                    'text': doc['page_content'],
                    'source': doc['metadata'].get('source', 'Unknown')
                }
            })
        return embedded_docs

    # Store to Pinecone
    def store_to_pinecone(embedded_docs):
        try:
            batch_size = 100
            for i in range(0, len(embedded_docs), batch_size):
                batch = embedded_docs[i:i+batch_size]
                index.upsert(vectors=batch)
        except Exception as e:
            st.error(f"Error uploading to Pinecone: {e}")

    # Embed and save documents
    embedded_docs = embed_documents(docs)
    store_to_pinecone(embedded_docs)

    # Clean up temporary file
    os.unlink(temp_file_path)

    return embeddings

# Retrieve relevant documents function
def retrieve_relevant_info(embeddings, query, top_k=5):
    query_embedding = embeddings.embed_query(query)
    
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results['matches']
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return []

# Generate response function
def generate_response(embeddings, query):
    relevant_docs = retrieve_relevant_info(embeddings, query)
    context = "\n".join([doc["metadata"]["text"] for doc in relevant_docs])

    prompt = f"""You are a financial analyst specializing in profit and loss statements. Based on the financial data provided, 
    answer the following question in a detailed and sentence-based format. Do not break up words or numbers. If the answer contains 
    numerical values, ensure they are presented properly with the correct formatting. The answer should not have any unnecessary 
    formatting.

    **Context:**
    {context}

    **Query:**
    {query}

    **Instructions:**
    - Provide a clear, well-structured answer with no markdown. If there is a numerical answer, explain the context behind the numbers (e.g., percentage increase, variance).
    - If the answer is numerical, explain the context behind the numbers (e.g., percentage increase, variance).
    - Keep the response concise but informative, focusing on key metrics.
    """

    model = genai.GenerativeModel("gemini-pro")
    
    # Define generation parameters to control randomness
    generation_config = {
        "temperature": 0.0,           # Lower temperature for more focused outputs
        "top_p": 0.8,                # Slightly restrictive top_p for more predictable text
        "top_k": 40,                 # Limit the token selection pool
        "max_output_tokens": 1024,    # Control response length
        "candidate_count": 1         # Generate single response instead of multiple
    }
    
    # Create safety settings to ensure professional responses
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]

    # Generate content with controlled parameters
    response = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    return response.text

# Main Streamlit app logic
def main():
    if uploaded_file and st.session_state.embeddings is None:
        # Add spinner during document upload and processing
        with st.spinner("Processing your document... This may take a few moments."):
            # Process PDF and create embeddings (run only once)
            embeddings = process_pdf(uploaded_file)
            st.session_state.embeddings = embeddings

        # Success message after the first upload
        st.success("Successfully uploaded and processed the document.")

    # Chat input
    if prompt := st.chat_input("Ask a question about the financial document"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
                # Generate response only if embeddings are loaded
                if st.session_state.embeddings:
                    response = generate_response(st.session_state.embeddings, prompt)
                else:
                    response = "Please upload a document first."
                st.write(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Run the app
if __name__ == "__main__":
    main()
