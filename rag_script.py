import numpy as np
import faiss
import fitz
from tqdm.auto import tqdm
from pathlib import Path
from spacy.lang.en import English
import re
import json
import os
from sentence_transformers import SentenceTransformer

class RAGPipeline:
    def __init__(self, model_name_or_path: str = "all-mpnet-base-v2", 
                 device: str = "cpu",
                 num_sentence_chunk_size: int = 10,
                 original_pageno: int = None):
        self.device = device
        self.model_name_or_path = model_name_or_path
        self.num_sentence_chunk_size = num_sentence_chunk_size
        self.embedding_model = SentenceTransformer(model_name_or_path=self.model_name_or_path, device=self.device)
        if original_pageno:
            self.original_pagenumber = original_pageno
        else:
            self.original_pagenumber = 1
        self.nlp = English()
        self.nlp.add_pipe('sentencizer')
        self.index = None
        self.metadata = []

    @staticmethod
    def text_formatter(text: str) -> str:
        cleaned_text = text.replace("\xa0", " ")  # Handles non-breaking space
        cleaned_text = cleaned_text.replace("\n", " ").strip()
        return cleaned_text

    @staticmethod
    def split_list(input_list: list, slice_size: int = 10) -> list[list[str]]:
        return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]

    def open_and_read_pdf(self, pdf_path: str | Path, original_page_number: int = None) -> list[dict]:
        doc = fitz.open(pdf_path)
        pages_and_texts = []
        for page_number, page in tqdm(enumerate(doc), total=len(doc)):
            text = page.get_text()
            text = self.text_formatter(text=text)
            pages_and_texts.append({
                "page_number": page_number - self.original_pagenumber,  
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split('.')),
                "text": text
            })
        return pages_and_texts

    def sentence_maker(self, pages_and_texts: list[dict]) -> list[dict]:
        for item in tqdm(pages_and_texts, total=len(pages_and_texts)):
            item['sentences'] = [str(sentence) for sentence in self.nlp(item['text']).sents]
            item['page_sentence_count_spacy'] = len(item['sentences'])

        for item in tqdm(pages_and_texts):
            item['sentence_chunks'] = self.split_list(input_list=item['sentences'], slice_size=self.num_sentence_chunk_size)
            item['num_chunks'] = len(item['sentence_chunks'])

        return pages_and_texts

    def pages_and_chunker(self, pages_and_texts: list[dict]) -> list[dict]:
        pages_and_chunks = []
        for item in tqdm(pages_and_texts):
            for sentence_chunk in item['sentence_chunks']:
                chunk_dict = {
                    'page_number': item['page_number'],
                    'sentence_chunk': "".join(sentence_chunk).replace("  ", " ").strip(),
                }

                # Handle edge cases in sentence formatting
                chunk_dict['sentence_chunk'] = re.sub(r'\.([A-Z])', r'. \1', chunk_dict['sentence_chunk'])
                chunk_dict['chunk_char_count'] = len(chunk_dict['sentence_chunk'])
                chunk_dict['chunk_word_count'] = len(chunk_dict['sentence_chunk'].split(' '))
                chunk_dict['chunk_token_count'] = len(chunk_dict['sentence_chunk']) / 4  # Estimation for token count

                pages_and_chunks.append(chunk_dict)
        return pages_and_chunks

    def embed_text(self, chunks: list[dict]) -> list[dict]:
        for chunk in tqdm(chunks, total=len(chunks)):
            chunk_text = chunk['sentence_chunk']
            chunk['embedding'] = self.embedding_model.encode(chunk_text)
        return chunks

    def process_pdf(self, pdf_path: str | Path):
        pages_and_texts = self.open_and_read_pdf(pdf_path)
        pages_and_texts = self.sentence_maker(pages_and_texts)
        chunks = self.pages_and_chunker(pages_and_texts)
        embedded_chunks = self.embed_text(chunks)
        return embedded_chunks
    
    def save_embeddings_to_faiss_with_metadata(self, embeddings: list[dict], index_file: str, metadata_file: str):
        vectors = np.array([chunk['embedding'] for chunk in embeddings], dtype=np.float32)

        dimension = vectors.shape[1]  # Embedding dimension size = 768
        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(vectors)

        faiss.write_index(self.index, index_file)

        self.metadata = [{'page_number': chunk['page_number'], 'sentence_chunk': chunk['sentence_chunk']} for chunk in embeddings]
        with open(metadata_file, 'w') as metadata_f:
            json.dump(self.metadata, metadata_f)
            
    def load_faiss_and_metadata(self, index_file: str, metadata_file: str):
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
        else:
            raise FileNotFoundError(f"FAISS index file {index_file} not found.")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as metadata_f:
                self.metadata = json.load(metadata_f)
        else:
            raise FileNotFoundError(f"Metadata file {metadata_file} not found.")
    def search_in_faiss(self, query:str, k: int = 5):
        if self.index is None:
            raise ValueError("FAISS index is not loaded. Call load_faiss_and_metadata first.")

        distances, indices = self.index.search(np.array(self.embed_query(query)), k)

        results = []
        for idx in indices[0]:
            result_metadata = self.metadata[idx]
            result_metadata['distance'] = distances[0][indices[0].tolist().index(idx)]  # Add distance for reference
            results.append(result_metadata)
        
        return results
    def embed_query(self, query: str):
        return np.array([self.embedding_model.encode(query, convert_to_numpy=True)])


# rag_pipeline = RAGPipeline(model_name_or_path="all-mpnet-base-v2", device="cpu", num_sentence_chunk_size=10)
# embedded_chunks = rag_pipeline.process_pdf("week 1.pdf")
# rag_pipeline.save_embeddings_to_faiss_with_metadata(embedded_chunks, "faiss_index.index", "metadata.json")
# rag_pipeline.load_faiss_and_metadata("faiss_index.index", "metadata.json")
# query = "Why UML model"
# results = rag_pipeline.search_in_faiss(query=query, k=5)



