## QA with multiple file formats

This repository contains the code for a Document Search and Question-Answering (QA) system implemented using LangChain, an open-source library for natural language processing tasks. The system allows users to provide a query, and it will search through a collection of documents in various formats (PDF, DOCX, TXT, and PPTX) to find relevant answers and provide the corresponding sources.

#### Getting Started
To use this system, follow the steps below:

1. Clone the repository to your local machine
2. Set up the OpenAI API key: To use the OpenAI-powered embeddings, you need to set your OpenAI API key as an environment variable. If you don't have one, you can sign up for the OpenAI API and get the key.
3. Install the required dependencies
4. Place your documents in the appropriate folder: Create a folder named Q&A with PDF & Txt in the project directory and place your documents (PDF, DOCX, TXT, and PPTX) inside this folder.

#### Usage
To start the chat loop and use the Document Search and QA system, run the following command:

`python Q&A with PDF, Txt & Docx files with OpenAI Embeddings.py`

You can now enter your queries and get answers based on the contents of the documents. Type exit to end the chat loop.

#### Supported Document Formats
The system supports the following document formats:

- PDF
- DOCX (Microsoft Word)
- TXT (Plain text)
- PPTX (Microsoft PowerPoint)


#### How It Works
1. Document Loading:
The system loads all the supported documents from the specified file paths and stores them for further processing.

2. Text Splitting:
To handle large documents efficiently, the system splits them into smaller chunks using the Recursive Character Text Splitter.

3. Document Embeddings:
The text chunks are embedded using OpenAI's embeddings to convert text into vector representations for efficient search.

4. Document Search:
The system uses Chroma from LangChain to perform vector-based search on the embedded text chunks.

5. Question-Answering:
The VectorDBQA from LangChain is used for question-answering. The system takes the user's query, searches through the documents, and returns relevant answers along with their sources and page numbers (if available).

#### Note
- Ensure that you have set up the OpenAI API key properly for the embeddings to work.
- Make sure that the required Python dependencies are installed before running the system.
- The system may work with additional document types by extending the document loaders accordingly.


Please note that this README is a general template, and you may need to modify it to match the actual structure of your repository and add more specific information about the LangChain library and other implementation details.
