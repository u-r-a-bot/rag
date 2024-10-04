import torch,re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,TextIteratorStreamer
from transformers.utils import is_flash_attn_2_available  
from rag_script import RAGPipeline  

class LLMModelHandler:
    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct", device='cpu'):
        """Initializes the LLMModelHandler by loading the tokenizer and model"""
        self.model_id = model_id
        self.device = device
        self.tokenizer = None
        self.llm_model = None
        self.quantization_config = None
        self.attn_implementation = None
        self.rag_pipeline = None
    def load_model(self):

        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        
        # Determine attention implementation
        if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
            self.attn_implementation = "flash_attention_2"
        else:
            self.attn_implementation = "sdpa"
        
        print(f"[INFO] Using attention implementation: {self.attn_implementation}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_id)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            torch_dtype=torch.float16,
            quantization_config=self.quantization_config,
            low_cpu_mem_usage=True,
            attn_implementation=self.attn_implementation
        )
    def build_faiss_index(self, pdf_file: str):
        """Processes a PDF and embeds it, then saves the FAISS index and metadata."""
        # Initialize RAGPipeline if it's not already initialized
        if not self.rag_pipeline:
            self.rag_pipeline = RAGPipeline(model_name_or_path="all-mpnet-base-v2", device=self.device, num_sentence_chunk_size=10)
        
        # Process the PDF and create embeddings
        print(f"Processing PDF: {pdf_file}")
        embedded_chunks = self.rag_pipeline.process_pdf(pdf_file)
        

        print(f"Saving FAISS index and metadata...")
        self.rag_pipeline.save_embeddings_to_faiss_with_metadata(embedded_chunks, "faiss_index.index", "metadata.json")
        self.index_built = True
        
    def load_faiss_index(self):
        if not self.index_built:
            print("FAISS index not built. Please call 'build_faiss_index' first.")
            return
        
        print(f"Loading FAISS index and metadata...")
        self.rag_pipeline.load_faiss_and_metadata("faiss_index.index", "metadata.json")

    def prompt_for_augmentation(self, query: str):

        if not self.index_built:
            print("[ERROR] FAISS index not built or loaded. Please call 'build_faiss_index' or 'load_faiss_index' first.")
            return

        print(f"Searching FAISS index for query: {query}")
        results = self.rag_pipeline.search_in_faiss(query=query, k=5)
        
        base_prompt = """Based on the following context please answer the query:
        context:- {context}
        query:- {query}
        """
        context = ""
        for x in results:
            context += x['sentence_chunk'] + '\n'
        

        base_prompt = base_prompt.format(context=context, query=query)
        dialogue_template = [{"role": "user", "content": base_prompt}]
        
        return dialogue_template

    def format_output(self , raw_text: str, query = '') -> str:
        formatted_text = re.sub(r'<\|begin_of_text\|>|<\|end_of_text\|>|<\|eot_id\|>', '', raw_text)

        formatted_text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', formatted_text)
        formatted_text = re.sub(r'Cutting Knowledge Date:.*?\n', '', formatted_text)
        formatted_text = re.sub(r'Today Date:.*?\n', '', formatted_text)
        formatted_text = re.sub(r'You are a smart AI assistant.*?\n', '', formatted_text)
        formatted_text = re.sub(r'\n+', '\n', formatted_text).strip()
        formatted_text = formatted_text.replace(query, '')
        formatted_text = formatted_text.strip()
        return formatted_text
    def generate_output(self, query: str, to_rag: bool = False, context: dict = None):
        """Generates output from the model either with or without RAG augmentation."""
        # Ensure the model and tokenizer are loaded
        if self.llm_model is None or self.tokenizer is None:
            self.load_model()
        
        dialogue_template = []
        
        # If RAG is enabled, augment the prompt with retrieved context
        if to_rag:
            dialogue_template = self.prompt_for_augmentation(query=query,)
        else:
            dialogue_template = [{"role": "system", "content": "You are a smart AI assistant"}]
            if context:
                dialogue_template.append(context)
            dialogue_template.append({"role": "user", "content": query})

        # Prepare the prompt for generation
        prompt = self.tokenizer.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(prompt, return_tensors='pt')


        outputs = self.llm_model.generate(
            **input_ids,
            temperature=0.7, 
            do_sample=True,
            max_new_tokens=256,
            
        )
        

        output_text = self.format_output(self.tokenizer.decode(outputs[0]), query = query)
        print(output_text)
        
        return output_text
    
    
    def generate_output_streaming(self, query: str, to_rag: bool, context=None):
        """Generates output from the model either with or without RAG augmentation."""

        if self.llm_model is None or self.tokenizer is None:
            self.load_model()
        
        dialogue_template = []
        

        if to_rag:
            dialogue_template = self.prompt_for_augmentation(query=query)
        else:
            dialogue_template = [{"role": "system", "content": "You are a smart AI assistant"}]
            if context:
                for x in context:
                    dialogue_template.append(x)
            dialogue_template.append({"role": "user", "content": query})

        prompt = self.tokenizer.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids']
        

        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id 
        num_tokens = 50

        for _ in range(256):  
            outputs = self.llm_model.generate(
                input_ids=input_ids,
                temperature=0.7,
                do_sample=True,
                max_new_tokens=num_tokens,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id
            )
            

            next_token_id = outputs[0, -num_tokens:]  
            next_token = self.tokenizer.decode(next_token_id, skip_special_tokens=True)

            yield next_token  

            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

            if eos_token_id in next_token_id or "<|endoftext|>" in next_token:
                break