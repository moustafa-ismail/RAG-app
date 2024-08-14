import gradio as gr
import os
import shutil
# Requires !pip install PyMuPDF, see: https://github.com/pymupdf/pymupdf
import fitz # (pymupdf, found this is better than pypdf for our use case, note: licence is AGPL-3.0, keep that in mind if you want to use any code commercially)
from tqdm.auto import tqdm # for progress bars, requires !pip install tqdm 
import pandas as pd
from spacy.lang.en import English # see https://spacy.io/usage for install instructions
import re
import numpy as np
from sentence_transformers import util, SentenceTransformer
import textwrap
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import pipeline
from threading import Thread
from langchain_huggingface.llms import HuggingFacePipeline
from RAG_app.model_config_v11082024 import embedding_model



# For formatting text and removing spaces
def text_formatter(text: str) -> str:
        """Performs minor formatting on text."""
        cleaned_text = text.replace("\n", " ").strip() # note: this might be different for each doc (best to experiment)

        # Other potential text formatting functions can go here
        return cleaned_text


# Open PDF and get lines/pages
# Note: this only focuses on text, rather than images/figures etc
def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    doc = fitz.open(pdf_path)  # open a document
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
        text = page.get_text()  # get plain text encoded as UTF-8
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number,  # adjust page numbers since our PDF starts on page 42
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                                "text": text})
    return pages_and_texts



# Split list function with overlap
def split_list_with_overlap(input_list: list, slice_size: int, overlap: int) -> list[list[str]]:
    result = []
    for i in range(0, len(input_list), slice_size - overlap):
        result.append(input_list[i:i + slice_size])
    return result


# Chunking
def chunking(my_pages_and_texts):

    # print(my_pages_and_texts)
    nlp = English()

    # Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/ 
    nlp.add_pipe("sentencizer")


    # Define split size to turn groups of sentences into chunks
    num_sentence_chunk_size = 5
    chunk_overlap = 1

    # splitted_list = split_list()
    

    # Loop through pages and texts and split sentences into chunks with overlap
    for item in tqdm(my_pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        item["sentence_chunks"] = split_list_with_overlap(input_list=item["sentences"],
                                                        slice_size=num_sentence_chunk_size,
                                                        overlap=chunk_overlap)
        item["num_chunks"] = len(item["sentence_chunks"])

    

    chunk_df = pd.DataFrame(my_pages_and_texts)
    chunk_sentence_stats_df = chunk_df.describe().round(2)

    # Split each chunk into its own item
    pages_and_chunks = []
    for item in tqdm(my_pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]
            
            # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo 
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get stats about the chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters
            
            pages_and_chunks.append(chunk_dict)

    # Important to check the token length of the embedding model
    df_pages_and_chunks= pd.DataFrame(pages_and_chunks)
    chunk_stats_df = df_pages_and_chunks.describe().round(2)


    # min_token_length = 30
    # pages_and_chunks_over_min_token_len = df_pages_and_chunks[df_pages_and_chunks["chunk_token_count"] > min_token_length]#.to_dict(orient="records")


    return chunk_stats_df, df_pages_and_chunks, pages_and_chunks


# Create a function that recursively splits a list into desired sizes 
def split_list(input_list: list, 
            slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible).

    For example, a list of 17 sentences would be split into two lists of [[10], [7]]
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]



def embed_doc(pages_and_chunks_over_min_token_len):

    pages_and_chunks_over_min_token_len_dict = pages_and_chunks_over_min_token_len.to_dict(orient="records")
    # Create embeddings one by one on the GPU
    for item in tqdm(pages_and_chunks_over_min_token_len_dict):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

    # Turn text chunks into a single list **** to use later in query_with_context ****
    text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len_dict]

    # print(text_chunks[:2])
    # Create embeddings one by one on the GPU
    for item in tqdm(pages_and_chunks_over_min_token_len_dict):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])   


    

    # Save embeddings to file
    text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len_dict)


    # Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
    # text_chunks_and_embeddings_df["embedding"] = text_chunks_and_embeddings_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    
    embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
    text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

   

    return text_chunks_and_embeddings_df.head(), text_chunks_and_embeddings_df, text_chunks



# Function that takes a query and chunks and returns a prompt with the query and context in format
def query_with_context(query, text_chunks_and_embeddings_df, pages_and_chunks, text_chunks):
    
    # Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
    text_chunks_and_embeddings_df["embedding"] = text_chunks_and_embeddings_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))    
    
    
    # Convert texts and embedding df to list of dicts
    pages_and_chunks = text_chunks_and_embeddings_df.to_dict(orient="records")
    embeddings = text_chunks_and_embeddings_df["embedding"]
    
    # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
    embeddings = torch.tensor(np.array(text_chunks_and_embeddings_df["embedding"].tolist()), dtype=torch.float32).to('cuda')
    

    text_chunk_embeddings = embedding_model.encode(text_chunks,
                                               batch_size=32,
                                               convert_to_tensor=True)
    
    # print("this is the type of embedding: ", type(text_chunk_embeddings))

    # print("this is the shape of embedding: ", text_chunk_embeddings.shape)
    print("query")
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    dot_scores = util.dot_score(a=query_embedding, b=text_chunk_embeddings)[0]

    # 4. Get the top-k results (we'll keep this to 5)
    top_results_dot_product = torch.topk(dot_scores, k=2)

    def print_wrapped(text, wrap_length=80):
        wrapped_text = textwrap.fill(text, wrap_length)
        return wrapped_text


    print_list = [f"Query: '{query}'\n", "Results: \n"]

    # for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
    #     print_list.append(f"Score: {score:.4f}\n")
    #     # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
    #     print_list.append("Text:\n")
    #     print_list.append(print_wrapped(pages_and_chunks[idx]["sentence_chunk"]) + "\n")
    #     # Print the page number too so we can reference the textbook further (and check the results)
    #     print_list.append(f"Page number: {pages_and_chunks[idx]['page_number']}")
        
    #     print_list.append("\n")
    #     print_list.append("\n")
        

    # print_text = "".join(print_list)
    print_text = ''
    # Correct the extraction and combination of top texts
    combined_prompt = " ".join([text_chunks[idx] for idx in top_results_dot_product.indices])
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX\nThe combined prompt:", combined_prompt)
    query_with_context = f"""System: You are a chatbot that will answer from provided context after understanding all of it and return the answer.
    Query: {query}
    Context: {combined_prompt}
    Answer:"""

    return query_with_context, print_text, combined_prompt


# # Predicting function
# def predict(messages, history, retrieved_context):

#     query = f"Query: {messages}\n"
#     context = f"Context: {retrieved_context}\nAnswer:"

#     def mypipe():
#     #    pipe(messages)
#         llm_chat(query + context, streamer=streamer)

#     thread = Thread(target=mypipe)
#     thread.start()
#     generated_text = ""
#     for new_text in streamer:
#         generated_text += new_text

#         yield generated_text


