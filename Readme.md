# RAG App

Welcome to the github repo: Hello World! (Mohey you are welcome to edit :p)

Retrieval Augmented Gen AI (RAG) with interface. 

Choose a model to load from huggingface and customize the interface to allow tweaking
parameters such as chunk size, embedding model, LLM parameters.

The interface using Gradio. (hope to add flask and streamlit)

The app should allow uploading documents to a folder or create a database to store embeddings
and documents.

Working on handling multiple documents
Working on handling history
Working on gui

I don't know what we are doing.

Tools Used: 
  1. Huggingface transformers for loading LLMs.
  2. Huggingface Sentence Transformers for loading embedding models.
  3. Langchain for creating pipeline.
  4. Huggingface Gradio for interface.
  5. Streamlit for interface.
  6. Flask for web gui.

<br>


### **Steps to run:**

1. Download this github repository
2. Create a virtual environment

     `python -m venv venv`
3. Activate the virtual environment

    `source venv/bin/activate` or `venv/bin/activate` [for Windows]

4. Install the requirements

   `pip install -r requirements.txt`

6. On the terminal run the command below 

     `python app.py`


<br>

<!-- <p align="center">
<img align="center" width="724" alt="Screen Shot 2024-01-11 at 1 49 47 PM" src="https://github.com/ashhass/Chatbot/assets/53818655/e6b2d942-0db5-4d40-b05a-b70d2b5fd042" >
</p> -->