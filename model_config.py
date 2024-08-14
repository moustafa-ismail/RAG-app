from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import pipeline
from langchain_huggingface.llms import HuggingFacePipeline
import torch
from tqdm import tqdm
import time
import gc
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig


# "dwzhu/e5-base-4k"
# "intfloat/multilingual-e5-large"
# "mixedbread-ai/mxbai-embed-large-v1"

def load_embed_model():

    embedding_model = SentenceTransformer(model_name_or_path="C:/Users/Dell 2/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-l6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a",
                                          device="cuda",
                                          local_files_only=True,
                                          )
    
    print("embedding is loaded \n \n \n")
    return embedding_model


embedding_model = load_embed_model()



def empty_gpu_cache():

    global model
    global hf_pipe
    global llm_chat
    global embedding_model

    del model
    del hf_pipe
    del llm_chat
    del embedding_model

    gc.collect()

    torch.cuda.empty_cache()

    # global embedding_model
    # del embedding_model

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    time.sleep(1)
    print("should be emptied now")

    return None


def load_model_into_pipeline(model_name_hf, empty_gpu_cache = True):
    '''
    Use huggingface model name and return pipelinem and its streamer to use
    
    '''
    # if empty_gpu_cache == True:
    # if if_model_loaded() == True:

    global model
    global hf_pipe
    global llm_chat

    quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    )


    tok = AutoTokenizer.from_pretrained(model_name_hf)
    # print("before model and pipeline")
    # print("Memory usage: ", torch.cuda.memory_allocated(0)/(1024*1024*1024))
    model = AutoModelForCausalLM.from_pretrained(model_name_hf, device_map = "cuda", 
                                                quantization_config=quantization_config,

                                                )
    
    streamer = TextIteratorStreamer(tok, skip_prompt=True)
    hf_pipe = pipeline(task="text-generation", 
                    model=model, 
                    tokenizer=tok,
                    return_full_text=False, 
                    streamer=streamer,
                    max_new_tokens=200
                    )
    

    llm_chat = HuggingFacePipeline(pipeline=hf_pipe)


    
    
    # print("after model and pipeline")
    # print("Memory usage: ", torch.cuda.memory_allocated(0)/(1024*1024*1024))

    return llm_chat, streamer, hf_pipe


# llm_model_pipeline, my_streamer = load_model_into_pipeline(model_name)



