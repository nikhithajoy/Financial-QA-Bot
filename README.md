# Interactive QA Bot Interface for Financial Data
This project provides an interactive QA bot that allows users to upload financial documents (e.g., Profit & Loss statements in PDF format) and query the data to receive structured, financial insights. The bot leverages advanced language models for analyzing documents and generating accurate, context-aware responses.

## Features
- Upload and process financial documents: Upload PDF documents containing Profit & Loss statements.
- Ask financial queries: Query the bot for insights into the uploaded financial data, such as revenue, expenses, profit margins, etc.
- Structured responses: Receive context-based answers that include numerical insights and explanations.
- Clear chat history: Reset the conversation at any time to start fresh.

## Prerequisites
Before running the application, make sure you have the following prerequisites installed:

- Python 3.8+
- Streamlit
- Pinecone
- Google Generative AI (Gemini)
- LlamaParse
- SentenceTransformers
- dotenv


## Required Python Libraries
To install the required libraries, run the following command:
~~~
pip install -r requirements.txt
~~~

## Environment Variables
The application uses several API keys to interact with external services. Make sure to set the following environment variables in a .env file:
~~~
pinecone_api_key=YOUR_PINECONE_API_KEY
gemini_api_key=YOUR_GOOGLE_GENERATIVE_AI_API_KEY
llama_key=YOUR_LLAMA_API_KEY
~~~

## Getting Started
### 1. Clone the Repository
Clone this repository to your local machine:
~~~
git clone https://github.com/yourusername/interactive-financial-qa-bot.git
cd interactive-financial-qa-bot
~~~
### 2. Set Up Environment
Create a .env file in the project root directory and add your API keys as mentioned in the Environment Variables section above.

### 3. Install Dependencies
Run the following command to install all required libraries:
~~~
pip install -r requirements.txt
~~~
### 4. Run the Streamlit App
Once the dependencies are installed and environment variables are set, you can run the application with the following command:
~~~
streamlit run app.py
~~~

## Usage
### Uploading Documents
Go to the sidebar and upload a PDF file containing your financial data (e.g., a Profit & Loss statement).
The bot will process the document and generate embeddings for querying.

### Asking Questions
After uploading, type your query in the chat input box, such as:
- "What was the total revenue last quarter?"
- "What is the variance in gross profit?"
- "List the main expense categories for Q2."
  
### Interpreting Responses
The bot will provide structured, concise answers. For example, if you ask about gross profit, the bot might respond with:
"The gross profit for the last quarter is $500,000, which represents a 40% margin on total revenue."

### Clear Chat History
To clear the chat and start a new conversation, click the "Clear Chat History" button in the sidebar.
