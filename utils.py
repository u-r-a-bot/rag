from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
from transformers.utils import is_flash_attn_2_available 
from transformers import BitsAndBytesConfig
import torch
from rag_script import RAGPipeline

def load_model():
    """ Loads the LLM model and the tokenizer
    """
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_compute_dtype=torch.float16)
    if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"
    print(f"[INFO] Using attention implementation: {attn_implementation}")
    
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                 torch_dtype=torch.float16, # datatype to use, we want float16
                                                 quantization_config=quantization_config ,
                                                 low_cpu_mem_usage=True, 
                                                 attn_implementation=attn_implementation,) 
    return tokenizer , llm_model


def prompt_for_augmentation(query:str, ):
    rag_pipeline = RAGPipeline(model_name_or_path="all-mpnet-base-v2", device="cpu", num_sentence_chunk_size=10)
    embedded_chunks = rag_pipeline.process_pdf("week 1.pdf")
    rag_pipeline.save_embeddings_to_faiss_with_metadata(embedded_chunks, "faiss_index.index", "metadata.json")
    rag_pipeline.load_faiss_and_metadata("faiss_index.index", "metadata.json")
    query = "Why UML model"
    results = rag_pipeline.search_in_faiss(query=query, k=5) # results is a dictionary with sentence_chunk as key for required input
    base_prompt = """ Based on the following context please answer the query
    context:- {context}
    query:- {query}
    """
    context = ""
    for x in results:
        context+= x['sentence_chunk'] 
        context+= '\n'
    base_prompt = base_prompt.format(context = context , query =query)
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]
    
    return dialogue_template

def generate_output(query:str , to_rag:bool, context = None):
    tokenizer, llm_model = load_model()
    dialogue_template = []
    if to_rag == True:
        dialogue_template = prompt_for_augmentation(query=query)
    else:
        dialogue_template = [
        {"role": "system", "content": "You are a smart AI assisstant"},
    ]
        if context:
            dialogue_template.append(context)
        dialogue_template.append({"role": "user",
        "content": query}
                                 )
    prompt = tokenizer.apply_chat_template(conversation = dialogue_template,tokenize=False,
                                          add_generation_prompt=True)
    input_ids = tokenizer(prompt , return_tensors = 'pt')
    outputs = llm_model.generate(**input_ids,
                             temperature=0.7, # lower temperature = more deterministic outputs, higher temperature = more creative outputs
                             do_sample=True, 
                             max_new_tokens=256) 
    output_text = tokenizer.decode(outputs[0])
    return output_text
    
    
