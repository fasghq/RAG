from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.document import ConversionResult
from docling.chunking import HybridChunker
from docling_core.types.doc import ImageRefMode
from docling.backend.msword_backend import MsWordDocumentBackend

from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader # TextLoader, Docx2txtLoader, PyPDFLoader
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Tuple, Dict
from pathlib import Path
import os
import json
from openai import OpenAI
from table_parser import process_file
from prompts import prompts
from config import OPENAI_API_KEY
from tqdm import tqdm

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = EasyOcrOptions(
    lang=['en', 'ru'],
    force_full_page_ocr=False
)
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
allowed_formats = [
    InputFormat.PDF,
    InputFormat.DOCX,
    InputFormat.MD,
    InputFormat.HTML,
    InputFormat.PPTX,
    InputFormat.CSV
]
format_options = {
    InputFormat.PDF: PdfFormatOption(
        pipeline_cls=StandardPdfPipeline,
        pipeline_options=pipeline_options,
        backend=DoclingParseV2DocumentBackend
    ),
    InputFormat.DOCX: WordFormatOption(
        pipeline_cls=SimplePipeline,
        pipeline_options=pipeline_options,
        backend=MsWordDocumentBackend
    )
}

converter = DocumentConverter(allowed_formats=allowed_formats, format_options=format_options)


client_gpt = OpenAI(
    api_key=OPENAI_API_KEY
)


def call_gpt(
    prompt_system: str, 
    prompt_user: str, 
    model: str = "gpt-4.1-mini", 
    max_tokens: int = 32768,
    temperature: float = 0.2
):
    request = client_gpt.responses.create(
        model=model,
        instructions=prompt_system,
        input=prompt_user,
        max_output_tokens=max_tokens,
        store=False,
        temperature=temperature
    )
    response = request.output_text
    return response


class RAG:
    def __init__(
        self,
        reranker_model_name: str = "DiTy/cross-encoder-russian-msmarco", # cross-encoder/ms-marco-MiniLM-L-12-v2
        embedding_model_name: str = "sergeyzh/rubert-mini-frida",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device

        # Initialize embeddings for initial retrieval
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': device}
        )

        # Initialize reranker
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
        self.reranker_model.to(device)

        # Initialize vector store
        self.vector_store = None

        # Initialize keyword retriever
        # for docs and doc chunks
        self.keyword_retriever_docs = None
        self.keyword_retriever_chunks = None 

        # Initialize ensemble retriever
        self.ensemble_retriever = None


    def load_and_split_documents(
        self,
        directory_path: str,
        chunk_size: int = 2048,
        chunk_overlap: int = 500
    ) -> List[Document]:
        """Load documents from a directory and split them into chunks."""
        documents = []
        chunks = {}
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        problem_files = []
        for filename in tqdm(os.listdir(directory_path)):
            file_path = os.path.join(directory_path, filename)
            if not os.path.isfile(file_path):
                continue
            ext = Path(filename).suffix.lower()
            if ext in [".txt", ".md"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if not content.strip():
                    problem_files.append(f"{filename}: Empty file")
                    continue
                doc = Document(page_content=content, metadata={"source": filename})
                doc_chunks = text_splitter.split_documents([doc])
            # elif ext == ".csv":
            #     loader = CSVLoader(
            #         file_path=file_path,
            #         encoding="utf-8-sig",
            #         # csv_args={},
            #     )
            #     pages = loader.load()
            #     content = "\n\n".join([page.page_content for page in pages])
            #     doc = Document(page_content=content, metadata={"source": filename})
            #     doc_chunks = text_splitter.split_documents(pages)
            elif ext in [".csv", ".xlsx", ".xls"]:
                content = process_file(Path(file_path))
                if not content.strip():
                    problem_files.append(f"{filename}: No tables")
                    continue
                doc = Document(page_content=content, metadata={"source": filename})
                doc_chunks = text_splitter.split_documents([doc])
            else:
                loader = DoclingLoader(
                    file_path=file_path,
                    converter=converter,
                    export_type=ExportType.MARKDOWN
                )
                pages = loader.load()
                if not pages:
                    problem_files.append(f"{filename}: No pages")
                    continue
                content = "\n\n".join([page.page_content for page in pages])
                if not content.strip():
                    problem_files.append(f"{filename}: Empty file")
                    continue
                doc = Document(page_content=content, metadata={"source": filename})
                doc_chunks = text_splitter.split_documents(pages)
            if doc_chunks:
                documents.append(doc)
                chunks[filename] = doc_chunks
            else:
                problem_files.append(f"{filename}: No chunks")
        print(problem_files)
        return documents, chunks
    

    def create_keywords_store_docs(self, documents: List[Document]):
        """Load a BM25 documents retriever."""
        self.keyword_retriever_docs = BM25Retriever.from_documents(documents)

    
    def create_keywords_store_chunks(self, chunks: Dict[str, List[Document]]):
        """Load a BM25 chunks retriever."""
        self.keyword_retriever_chunks = {}
        for filename, documents in tqdm(chunks.items()):
            self.keyword_retriever_chunks[filename] = BM25Retriever.from_documents(documents)


    def save_documents(self, documents: List[Document], documents_dir: str):
        os.makedirs(documents_dir, exist_ok=True)
        documents_file = os.path.join(documents_dir, "documents.json")
        with open(documents_file, "w", encoding="utf-8") as f:
            json.dump([doc.model_dump() for doc in documents], f, ensure_ascii=False, indent=2)


    def load_documents(self, documents_dir: str) -> List[Document]:
        documents_file = os.path.join(documents_dir, "documents.json")
        with open(documents_file, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)
        return [Document(**doc) for doc in raw_docs]
    

    def save_chunks(self, chunks: Dict[str, List[Document]], chunks_dir: str):
        os.makedirs(chunks_dir, exist_ok=True)
        for filename, chunks in chunks.items():
            file_chunks_dir = os.path.join(chunks_dir, filename)
            os.makedirs(file_chunks_dir, exist_ok=True)
            chunks_file = os.path.join(file_chunks_dir, "chunks.json")
            with open(chunks_file, "w", encoding="utf-8") as f:
                json.dump([chunk.model_dump() for chunk in chunks], f, ensure_ascii=False, indent=2)


    def load_chunks(self, chunks_dir: str) -> Dict[str, List[Document]]:
        chunks = {}
        for filename in os.listdir(chunks_dir):
            file_chunks_dir = os.path.join(chunks_dir, filename)
            if os.path.isdir(file_chunks_dir):
                chunks_file = os.path.join(file_chunks_dir, "chunks.json")
                if os.path.isfile(chunks_file):
                    with open(chunks_file, "r", encoding="utf-8") as f:
                        chunk_data = json.load(f)
                        chunks[filename] = [Document(**chunk) for chunk in chunk_data]
        return chunks


    def create_vector_store(self, chunks: Dict[str, List[Document]]):
        """Create a FAISS vector store from documents."""
        self.vector_store = {}
        for filename, documents in tqdm(chunks.items()):
            self.vector_store[filename] = FAISS.from_documents(documents, self.embeddings)


    def save_vector_store(self, vectors_dir: str):
        """Dumps a FAISS vector store to a given directory."""
        os.makedirs(vectors_dir, exist_ok=True)
        for filename, vector_store in self.vector_store.items():
            backup_path = os.path.join(vectors_dir, filename)
            os.makedirs(backup_path, exist_ok=True)
            vector_store.save_local(backup_path)


    def load_vector_store(self, dir: str):
        """Load a FAISS vector store from a local directory."""
        self.vector_store = {}
        for subdir in os.listdir(dir):
            backup_path = os.path.join(dir, subdir)
            if os.path.isdir(backup_path):
                try:
                    vector_store = FAISS.load_local(
                        backup_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    self.vector_store[subdir] = vector_store
                except Exception as e:
                    print(f"Failed to load vector store from {subdir}: {e}")
                    continue


    def expand_query(
        self, 
        query: str, 
        model: str = "gpt-4.1",
        max_tokens: int = 32768,
        temperature: float = 0.2,
        browsing: bool = False
    ) -> str:
        """Expand the query using LLM."""
        mode = "expand query web" if browsing else "expand query"
        prompt_user = f"Original query: {query}"
        expanded_query = call_gpt(
            prompt_system=prompts[mode], 
            prompt_user=prompt_user, 
            model=model, 
            max_tokens=max_tokens,
            temperature=temperature
        )
        return expanded_query.strip()


    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Rerank documents using the cross-encoder model."""
        pairs = []
        for doc in documents:
            pairs.append((query, doc.page_content))

        features = self.reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self.reranker_model(**features).logits.squeeze().cpu() # .numpy()

        # if isinstance(logits, (int, float)) or (hasattr(logits, "ndim") and logits.ndim == 0):
        #     scores = [float(logits)]
        if logits.ndim == 0:
            scores = [float(logits.item())]
        else:
            scores = [float(x) for x in logits.numpy().tolist()]

        doc_score_pairs = list(zip(documents, scores))
        ranked_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        return ranked_docs[:top_k]
    

    def evaluate_relevance(
        self, 
        original_query: str,
        documents: List[Tuple[Document, float]],
        model: str = "gpt-4.1",
        max_tokens: int = 32768,
        temperature: float = 0.2
    ) -> bool:
        """Evaluate relevance of collected documents using LLM."""
        if not documents:
            return False
        context = "\n\n".join([doc.page_content for doc, _ in documents])
        prompt_user = f"Original query: {original_query}\n\nContext:\n{context}"
        response = call_gpt(
            prompt_system=prompts["evaluate relevance"], 
            prompt_user=prompt_user, 
            model=model, 
            max_tokens=max_tokens,
            temperature=temperature
        )
        print("relevance llm respnce:", response)
        return "Yes" in response.strip().lower()
    

    def refine_query(
        self, 
        original_query: str, 
        failed_query: str, 
        documents: List[Tuple[Document, float]],
        model: str = "gpt-4.1",
        max_tokens: int = 32768,
        temperature: float = 0.2
    ) -> str:
        """Generate better search query based on previous results."""
        context = "\n\n".join([doc.page_content for doc, _ in documents])
        prompt_user = prompts["refine query"].format(
            original_query=original_query,
            failed_query=failed_query,
            context=context
        )
        new_query = call_gpt(
            prompt_system="", 
            prompt_user=prompt_user, 
            model=model, 
            max_tokens=max_tokens,
            temperature=temperature
        )
        return new_query.strip()


    def generate_response(
        self, 
        query: str, 
        documents: List[Tuple[Document, float]], 
        model: str = "gpt-4.1",
        max_tokens: int = 32768,
        temperature: float = 0.2,
        browsing: bool = False
    ) -> str:
        """Generate a response using LLM based on retrieved documents."""
        mode = "respond query web" if browsing else "respond query"
        context = "\n\n".join([doc.metadata['source'] + ":\n" + doc.page_content for doc, _ in documents])
        prompt_user = f"Context:\n{context}\n\nQuery: {query}\n"
        response = call_gpt(
            prompt_system=prompts[mode], 
            prompt_user=prompt_user, 
            model=model, 
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.strip()


    def process_query(
        self,
        query: str,
        max_attempts: int = 1,
        query_expansion: bool = False,
        keyword_docs_top_k: int = 2,
        keyword_chunks_top_k: int = 20,
        vector_top_k: int = 15,
        rerank_top_k: int = 10,
        weights: list = [0.4, 0.6],
        model: str = "gpt-4.1",
        max_tokens: int = 32768,
        temperature: float = 0.2,
        browsing: bool = False,
    ) -> dict:
        """
        Complete pipeline: query expansion -> retrieval -> reranking -> response generation
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Please load documents first.")
        
        original_query = query
        current_query = query
        attempted_queries = []

        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1}/{max_attempts} with query: '{current_query}'")
            attempted_queries.append(current_query)

            # Step 1: Query expansion only for 1st iteration
            print("query expanded")
            if attempt == 0 and query_expansion:
                search_query = self.expand_query(
                    query=current_query, 
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    browsing=browsing
                )
            else:
                search_query = current_query

            # Step 2: Keyword search for appropriate documents
            print("keyword search through docs completed")
            if keyword_docs_top_k > 0:
                self.keyword_retriever_docs.k = keyword_docs_top_k
                # initial_docs = self.keyword_retriever_docs.get_relevant_documents(search_query)
                initial_docs = self.keyword_retriever_docs.invoke(search_query)
                initial_docs_names = [doc.metadata['source'] for doc in initial_docs]
            else:
                initial_docs_names = list(self.vector_store.keys())

            # Step 3: Hybrid search through selected documents
            print("hybrid search inside docs completed")
            initial_chunks = []
            for doc in initial_docs_names:
                if doc in self.vector_store:
                    vector_retriever = self.vector_store[doc].as_retriever(search_kwargs={"k": vector_top_k})
                    self.keyword_retriever_chunks[doc].k = keyword_chunks_top_k
                    self.ensemble_retriever = EnsembleRetriever(
                        retrievers=[self.keyword_retriever_chunks[doc], vector_retriever],
                        weights=weights
                    )
                    # initial_chunks.extend(self.ensemble_retriever.get_relevant_documents(search_query))
                    initial_chunks.extend(self.ensemble_retriever.invoke(search_query))
        
            # Step 4: Rerank selected chunks
            print("chunks reranked")
            reranked_chunks = self.rerank_documents(search_query, initial_chunks, top_k=rerank_top_k)

            # Step 5: Evaluate relevance & generate response
            print("evaluated relevance")
            is_relevant = self.evaluate_relevance(
                original_query=original_query,
                documents=reranked_chunks,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            print(reranked_chunks, is_relevant)
            if is_relevant or (attempt == max_attempts - 1):
                print("Found relevant context or it's the last attempt. Generating response.")
                final_response = self.generate_response(
                    query=original_query, 
                    documents=reranked_chunks, 
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    browsing=browsing
                )
                return {
                    "original_query": original_query,
                    "response": final_response,
                    "retrieved_documents": reranked_chunks,
                    "attempted_queries": attempted_queries,
                    "final_query": current_query
                }

            # Step 6: Refine current query
            print("refined query")
            current_query = self.refine_query(
                original_query=original_query, 
                failed_query=current_query, 
                documents=reranked_chunks,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )

        return {
            "original_query": original_query,
            "response": "Не удалось найти релевантную информацию после нескольких попыток.",
            "retrieved_documents": [],
            "attempted_queries": attempted_queries,
            "final_query": current_query
        }
    

# agent = RAG()
# print("Start parsing files.")
# documents, chunks = agent.load_and_split_documents("agents/museum/files")

# print("Start saving files.")
# agent.save_documents(documents, "agents/museum/documents")
# agent.save_chunks(chunks, "agents/museum/chunks")

# print("Start creating stores.")
# agent.create_keywords_store_docs(documents)
# agent.create_keywords_store_chunks(chunks)
# agent.create_vector_store(chunks)

# print("Start saving vector store.")
# agent.save_vector_store("agents/museum/vectors")

# # # # alternatively load from backup:
# # # # rag_system.load_vector_store("agents/museum/vectors")

# print("Start answering query.")
# query = "Расскажи про эффективные практики на базе технологий искусственного интеллекта в «умном городе»."
# results = agent.process_query(query)
# print(results["response"])
