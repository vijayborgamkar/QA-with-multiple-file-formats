from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
import pickle
import textwrap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
import os
import warnings
warnings.filterwarnings("ignore")

# Set up the environment variable for the OpenAI API key
os.environ["OPENAI_API_KEY"] = ""


def get_documents(file_paths):
    documents = []

    for file_path in file_paths:
        _, extension = os.path.splitext(file_path.lower())
        if '.docx' in extension or '.doc' in extension:
            docx_loader = UnstructuredWordDocumentLoader(file_path)
            docx_document = docx_loader.load()
            documents += docx_document
        elif '.pdf' in extension:
            pdf_loader = PyPDFLoader(file_path)
            pdf_document = pdf_loader.load()
            documents += pdf_document
        elif '.txt' in extension:
            txt_loader = TextLoader(file_path)
            txt_document = txt_loader.load()
            documents += txt_document
        elif '.pptx' in extension:
            pptx_loader = UnstructuredPowerPointLoader(file_path)
            pptx_document = pptx_loader.load()
            documents += pptx_document
        elif '.docx' in extension or '.doc' in extension and '.pdf' in extension:
            docx_loader = UnstructuredWordDocumentLoader(file_path)
            docx_document = docx_loader.load()
            documents += docx_document
            pdf_loader = PyPDFLoader(file_path)
            pdf_document = pdf_loader.load()
            documents += pdf_document
        elif '.docx' in extension or '.doc' in extension and '.txt' in extension:
            docx_loader = UnstructuredWordDocumentLoader(file_path)
            docx_document = docx_loader.load()
            documents += docx_document
            txt_loader = TextLoader(file_path)
            txt_document = txt_loader.load()
            documents += txt_document
        elif '.docx' in extension or '.doc' in extension and '.pptx' in extension:
            docx_loader = UnstructuredWordDocumentLoader(file_path)
            docx_document = docx_loader.load()
            documents += docx_document
            pptx_loader = UnstructuredPowerPointLoader(file_path)
            pptx_document = pptx_loader.load()
        elif '.pdf' in extension and '.txt' in extension:
            pdf_loader = PyPDFLoader(file_path)
            pdf_document = pdf_loader.load()
            documents += pdf_document
            txt_loader = TextLoader(file_path)
            txt_document = txt_loader.load()
            documents += txt_document
        elif '.pdf' in extension and '.pptx' in extension:
            pdf_loader = PyPDFLoader(file_path)
            pdf_document = pdf_loader.load()
            documents += pdf_document
            pptx_loader = UnstructuredPowerPointLoader(file_path)
            pptx_document = pptx_loader.load()
            documents += pptx_document
        elif '.pdf' in extension and '.docx' in extension or '.doc' in extension:
            pdf_loader = PyPDFLoader(file_path)
            pdf_document = pdf_loader.load()
            documents += pdf_document
            docx_loader = UnstructuredWordDocumentLoader(file_path)
            docx_document = docx_loader.load()
            documents += docx_document
        elif '.txt' in extension and '.pptx' in extension:
            txt_loader = TextLoader(file_path)
            txt_document = txt_loader.load()
            documents += txt_document
            pptx_loader = UnstructuredPowerPointLoader(file_path)
            pptx_document = pptx_loader.load()
            documents += pptx_document
        elif '.pdf' in extension and '.docx' in extension or '.doc' in extension and'.pptx' in extension:
            pdf_loader = PyPDFLoader(file_path)
            pdf_document = pdf_loader.load()
            documents += pdf_document
            docx_loader = UnstructuredWordDocumentLoader(file_path)
            docx_document = docx_loader.load()
            documents += docx_document
            pptx_loader = UnstructuredPowerPointLoader(file_path)
            pptx_document = pptx_loader.load()
            documents += pptx_document
        elif '.txt' in extension and '.docx' in extension or '.doc' in extension and'.pptx' in extension:
            txt_loader = TextLoader(file_path)
            txt_document = txt_loader.load()
            documents += txt_document
            docx_loader = UnstructuredWordDocumentLoader(file_path)
            docx_document = docx_loader.load()
            documents += docx_document
            pptx_loader = UnstructuredPowerPointLoader(file_path)
            pptx_document = pptx_loader.load()
            documents += pptx_document
        elif '.txt' in extension and '.docx' in extension or '.doc' in extension and'.pdf' in extension:
            txt_loader = TextLoader(file_path)
            txt_document = txt_loader.load()
            documents += txt_document
            docx_loader = UnstructuredWordDocumentLoader(file_path)
            docx_document = docx_loader.load()
            documents += docx_document
            pdf_loader = PyPDFLoader(file_path)
            pdf_document = pdf_loader.load()
            documents += pdf_document
        elif '.txt' in extension and '.pptx' in extension and'.pdf' in extension:
            txt_loader = TextLoader(file_path)
            txt_document = txt_loader.load()
            documents += txt_document
            pptx_loader = UnstructuredPowerPointLoader(file_path)
            pptx_document = pptx_loader.load()
            documents += pptx_document
            pdf_loader = PyPDFLoader(file_path)
            pdf_document = pdf_loader.load()
            documents += pdf_document
        elif '.docx' in extension or '.doc' in extension and '.pdf' in extension and '.txt' in extension and '.pptx' in extenstion:
            docx_loader = UnstructuredWordDocumentLoader(file_path)
            docx_document = docx_loader.load()
            documents += docx_document
            pdf_loader = PyPDFLoader(file_path)
            pdf_document = pdf_loader.load()
            documents += pdf_document
            txt_loader = TextLoader(file_path)
            txt_document = txt_loader.load()
            documents += txt_document
            pptx_loader = UnstructuredPowerPointLoader(file_path)
            pptx_document = pptx_loader.load()
            documents += pptx_document

    return documents

def get_query_result(query, documents):
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Query documents
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
    result = qa({"query": query})

    result_text = result['result'].strip()
    source = result.get('source_documents', [{}])[0].metadata.get('source', '')
    page = result.get('source_documents', [{}])[0].metadata.get('page', '')

    return result_text, source, page

def chat_loop(file_paths):
    documents = get_documents(file_paths)
    if not documents:
        print("No supported files found.")
        return

    while True:
        query = input("Enter your query (type 'exit' to end): ")
        if query.lower() == 'exit':
            break

        result = get_query_result(query, documents)

        if result is not None:
            result_text, source, page = result
            print("Result:", result_text)
            if source:
                print("Source:", source)
                print("Page:", page)
        else:
            print("No answer found for the query.")

        print()  # Print an empty line for separation

# Get the file paths from the webpage or wherever you have them
file_paths = [
    r'\Q&A with PDF & Txt\Documents\2022_Crime_prediction_using_a_hybrid_sentiment_analysis_approach_10.pdf',
    r'\Q&A with PDF & Txt\Documents\2015_Crime_prediction_using_twitter_sentiment_and_weather_11.pdf',
    r'\Q&A with PDF & Txt\Documents\ChatGPT and DALL-E for Text to Image conversion.pptx',
    r'\Q&A with PDF & Txt\Documents\All_News_with_NER_RE.txt',
    r'\Q&A with PDF & Txt\Documents\Data Scientist.docx'
]

# Start the chat loop
chat_loop(file_paths)