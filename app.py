import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import pipeline
from threading import Thread
import tqdm
from RAG_app.model_config_v11082024 import empty_gpu_cache, load_model_into_pipeline
from RAG_app.rag_from_scratch_functions_v11082024 import *
from tqdm.auto import tqdm
from pathlib import Path, PureWindowsPath

last_embedding_model = "intfloat/multilingual-e5-large"
model_name = "openai-community/gpt2"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# if gr.NO_RELOAD:
# global c
# c = 0





def call_model_load(model_name, progress=gr.Progress(track_tqdm=True)):
    # Calls the model_load_into_pipeline and returns the pipeline and streamer

    global llm_chat
    global streamer
    global hf_pipe

    # model_name_current = model_name
    # # model_name_previous = model_name_current
    # print(model_name)

    
    llm_chat, streamer, hf_pipe = load_model_into_pipeline(model_name_hf=model_name)


    return None
    
def empty_gpu():  

    global hf_pipe
    global llm_chat

    del hf_pipe
    del llm_chat

    empty_gpu_cache()
    gr.Info("Should be emptied now")
    return None



# Process pdf into dictionary of pages
def process_doc(pdf_path):
    
    docs_paths = os.listdir(pdf_path)
    # Open PDF and get lines/pages
    # Note: this only focuses on text, rather than images/figures etc
    for doc_path in docs_paths:
        the_pages_and_texts = open_and_read_pdf(pdf_path=doc_path)

    df = pd.DataFrame(the_pages_and_texts)
    # Get stats
    my_df = df.describe().round(2)


   
    # Chunking
    my_chunk_stats_df, my_df_pages_and_chunks, my_pages_and_chunks = chunking(the_pages_and_texts)
    
    

    # Embed function
    my_text_chunks_and_embeddings_df_head, my_text_chunks_and_embeddings_df_2, my_text_chunks = embed_doc(my_df_pages_and_chunks)
    
    gr.Info("Document processed successfully")
    return my_text_chunks_and_embeddings_df_head, my_text_chunks_and_embeddings_df_2, my_text_chunks


# Upload file function
# list_of_bytes [fileobj1, fileobj2, ...]
def upload_file(list_of_bytes):
    # get current path and append doc_upload to it
    path = Path.cwd().joinpath("doc_upload")

    # Remove the path if it already exists
    if os.path.exists(path):

        shutil.rmtree(path)

    # Create a new directory doc_upload
    os.mkdir(str(path))
    file_paths = [file.name for file in list_of_bytes]

    for fileobj in list_of_bytes:
        shutil.copy(fileobj, str(path))

    tmp_file_paths_string = '\n'.join(file_paths)
    
    
    return file_paths, str(path) + "/"



# Predict or infere from model
def predict(messages, history, text_chunks_emb_df, pages_and_chunks_dict, text_chunks_list):
    print("*******************")
    print(type(text_chunks_emb_df))
    print(type(pages_and_chunks_dict))
    print(type(text_chunks_list))
    print("*******************")
    print(messages)
    my_query_with_context, my_print_text, my_combined_prompt = query_with_context(messages, text_chunks_emb_df, pages_and_chunks_dict, text_chunks_list)
    Llama_message = [
    
        {"role": "system", "content": "You are a chatbot that will answer from provided context after understanding all of it well and return the direct answer based on the context and query"},
        {"role": "user", "content": f"""{messages}"""},
        {"role": "context", "content": f"""{my_combined_prompt}"""},


    ]

    

    def mypipe():
    #    pipe(messages)
        hf_pipe(Llama_message, streamer=streamer)
        # llm_chat(my_query_with_context, streamer=streamer)


    thread = Thread(target=mypipe)
    thread.start()
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text

        yield generated_text





with gr.Blocks(title='RAG',
        css=".contain { display: flex !important; flex-direction: column !important; }"
    "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
    "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
    "#col { height: calc(100vh - 16px) !important; }"
) as demo:
    
    # with gr.Row():
    gr.HTML(f"hello")
    
    # gr.Markdown("<img src='/file=click_its_logo.jpg'>")

    with gr.Row(equal_height=False):
        gr.Markdown("# Gen AI")
    with gr.Row(equal_height=False):
        with gr.Column(scale=3, ): 
            # gr.Markdown("<img src='/file=click_its_logo.jpg'>")
            # gr.Markdown("Ask Bella")

            # file upload

            # The upload_file is the function using gr.Uploadbutton() as input (returning list of bytes objects/strings) and gr.File() as output returning the files paths 
            file_output = gr.File(file_count="multiple", file_types=[".pdf"])
            pdf_textbox = gr.Textbox(label="Document Path")
            up_btn = gr.UploadButton("Click to Upload a File", file_types=[".pdf"], file_count="multiple")
            up_btn.upload(fn=upload_file, inputs=up_btn, outputs=[file_output, pdf_textbox])
            # pdf_out = gr.Textbox()


            # Process pdf outputs embedded chunks
            process_pdf_in = pdf_textbox
            process_pdf_out_1 = gr.DataFrame(visible=False) # df.head()
            process_pdf_out_2 = gr.DataFrame(visible=False) # text_chunks_and_embeddings_df
            process_pdf_out_3 = gr.Dropdown(visible=False) # text_chunks (list)
            # process_pdf_out_2 = gr.JSON(visible=False)
            process_pdf_btn = gr.Button(value="Process pdf")
            process_pdf_btn.click(process_doc, process_pdf_in, [process_pdf_out_1, 
                                                                process_pdf_out_2,
                                                                process_pdf_out_3])

            # extra_input = gr.Textbox()
            # extra_output = gr.Textbox(visible=False)
            # message_btn = gr.Button("extra message")
            # message_btn.click(extra_message, extra_input, extra_output)


            # Choose the model
            load_model_int = gr.Interface(
                call_model_load,

                inputs= gr.Dropdown(
            ["openai-community/gpt2", 
             "meta-llama/Meta-Llama-3.1-8B-Instruct",
             "TinyLlama/TinyLlama-1.1B-Chat-v1.0"], label="LLM model", info="Will add more models later!"
            ),

            outputs=None,

            submit_btn= "Load the model",
            show_progress="full"

            )


            # Empty GPU cache
            gpu_btn = gr.Button(value="Empty GPU cache")
            gpu_btn.click(empty_gpu, inputs=None, outputs=None)



        with gr.Column(scale=7, elem_id='col'):
            _ = gr.ChatInterface(
                predict,
                chatbot=gr.Chatbot(
                    show_copy_button=True,
                    render=False,
                    elem_id="chatbot",
                ),
                additional_inputs=[process_pdf_out_1,
                                   process_pdf_out_2,
                                   process_pdf_out_3],
                autofocus=False,
                show_progress="full"
            )

            # with gr.Accordion("see details"):
            #     gr.Markdown("lorem ipsum")


demo.launch(
    allowed_paths=["D:/gen-ai"],
    share=False,
    debug=True,
)











##################################################################################################



# def image(text):
#     html = (
#             "<div >"
#             "<img  src='file/click_its_logo.jpg' alt='click_its_logo.jpg'>"
#             + "</div>"
#     )
#     return html


