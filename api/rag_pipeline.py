import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile
from typing import List, Dict, Any
import os

class RAGPipeline:
    def __init__(self):
        # Initialize ChromaDB in-memory for Vercel deployment
        self.client = chromadb.Client(Settings(
            is_persistent=False,  # Use in-memory storage
            anonymized_telemetry=False
        ))
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Store collections in memory
        self.collections = {}

    async def process_text(self, text: str, metadata: Dict[str, Any]) -> str:
        """Process text content and store in ChromaDB"""
        try:
            # Split text into chunks
            splits = self.text_splitter.split_text(text)
            
            # Generate collection name from metadata
            collection_name = f"doc_{hash(str(metadata))}"
            
            # Create collection
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=metadata
            )
            
            # Add documents to collection
            ids = [f"id_{i}" for i in range(len(splits))]
            metadatas = [metadata for _ in splits]
            
            collection.add(
                documents=splits,
                metadatas=metadatas,
                ids=ids
            )
            
            # Store collection reference
            self.collections[collection_name] = collection
            
            return collection_name
            
        except Exception as e:
            raise Exception(f"Error processing text: {str(e)}")

    async def retrieve_relevant_chunks(self, 
                                    query: str, 
                                    collection_name: str, 
                                    n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from the collection"""
        if collection_name not in self.collections:
            raise Exception(f"Collection {collection_name} not found")
            
        collection = self.collections[collection_name]
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return [{
            'text': doc,
            'metadata': meta
        } for doc, meta in zip(results['documents'][0], results['metadatas'][0])]

    def get_context_string(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Convert relevant chunks into a formatted context string"""
        return "\n\n".join([
            f"Context {i+1}:\n{chunk['text']}"
            for i, chunk in enumerate(relevant_chunks)
        ])


