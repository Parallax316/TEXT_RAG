import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict, Any, Optional, Tuple
import requests
import json
import time
from tqdm import tqdm
import os
from llama_cpp import Llama
from dotenv import load_dotenv
import hashlib
from pathlib import Path

from backend.app.core.config import settings
from backend.app.services.vstore_svc import VectorStoreService
from backend.app.core.exceptions import DocumentProcessingError, EmbeddingError, RetrievalError, LLMError

# Load environment variables
load_dotenv()

class ClusterAnalysisService:
    """
    Service for analyzing document collections using clustering and LLM summarization.
    - Retrieves embeddings from ChromaDB
    - Clusters documents using DBSCAN
    - Generates summaries for each cluster using Phi-3-mini
    - Creates an overall collection summary
    """
    
    def __init__(self):
        self.vector_store_service = VectorStoreService()
        self.backend_url = os.environ.get("BACKEND_API_URL", "http://localhost:8000")
        
        # Initialize cache directory
        self.cache_dir = Path("cache/analysis")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Phi-3-mini model
        try:























































            
            self.llm = Llama.from_pretrained(
                repo_id="MoMonir/Phi-3-mini-128k-instruct-GGUF",
                filename="phi-3-mini-128k-instruct.Q6_K.gguf",
            )
        except Exception as e:
            print(f"Error loading Phi-3-mini model: {e}")
            raise
        
    def _chunk_text(self, text: str, max_chunk_size: int = 400) -> List[str]:
        """Split text into smaller chunks to fit within token limits."""
        # Simple sentence-based chunking
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            # Rough estimate of tokens (words + punctuation)
            sentence_size = len(sentence.split())
            if current_size + sentence_size > max_chunk_size:
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks

    def _call_llm(self, prompt: str, max_tokens: int = 100) -> str:
        """Call Phi-3-mini model with error handling and chunking"""
        try:
            # Ensure prompt fits within context window
            if len(prompt.split()) > 400:  # Rough estimate of tokens
                chunks = self._chunk_text(prompt)
                responses = []
                for chunk in chunks:
                    response = self.llm(
                        chunk,
                        max_tokens=max_tokens,
                        temperature=0.7,
                        stop=["</s>", "Human:", "Assistant:"],
                        echo=False
                    )
                    responses.append(response['choices'][0]['text'].strip())
                return " ".join(responses)
            else:
                response = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    stop=["</s>", "Human:", "Assistant:"],
                    echo=False
                )
                return response['choices'][0]['text'].strip()
        except Exception as e:
            raise LLMError(f"Error calling Phi-3-mini model: {e}")

    def _analyze_domain(self, documents: List[str]) -> str:
        """
        Analyzes a sample of documents to determine the domain and key themes.
        """
        try:
            # Take a sample of documents for domain analysis
            sample_size = min(3, len(documents))
            sample_docs = documents[:sample_size]
            
            # Process each document separately
            domain_analyses = []
            for i, doc in enumerate(sample_docs):
                prompt = f"""Human: Analyze this document and identify:
1. Main field of study
2. Specific domain
3. Key terms
4. Content type

Document:
{doc}

Assistant: Here's the analysis:"""
                
                analysis = self._call_llm(prompt, max_tokens=150)
                domain_analyses.append(analysis)
            
            # Combine analyses
            combined_prompt = f"""Human: Combine these domain analyses into one coherent analysis:

{chr(10).join([f'Analysis {i+1}: {analysis}' for i, analysis in enumerate(domain_analyses)])}

Assistant: Here's the combined analysis:"""
            
            return self._call_llm(combined_prompt, max_tokens=150)
                
        except Exception as e:
            print(f"Error analyzing domain: {e}")
            return "Unknown domain"

    def _summarize_cluster(self, cluster_docs: List[Dict[str, Any]], cluster_id: int, domain_analysis: str) -> str:
        """
        Generates a brief summary for a cluster using Phi-3-mini.
        """
        try:
            # Take only first 2 documents for summary
            sample_docs = cluster_docs[:2]
            summaries = []
            
            for doc in sample_docs:
                # Process each document separately
                prompt = f"""Human: Provide a one-sentence summary of this document:

{doc['content']}

Focus on the main topic or finding.

Assistant: Here's the summary:"""
                
                summary = self._call_llm(prompt, max_tokens=100)
                summaries.append(summary)
            
            # Combine summaries if needed
            if len(summaries) > 1:
                combined_prompt = f"""Human: Combine these summaries into one concise sentence:

Summary 1: {summaries[0]}
Summary 2: {summaries[1]}

Assistant: Here's the combined summary:"""
                return self._call_llm(combined_prompt, max_tokens=100)
            else:
                return summaries[0]
                
        except Exception as e:
            print(f"Error summarizing cluster {cluster_id}: {e}")
            return f"Error: Could not generate summary for cluster {cluster_id}"

    def _generate_overall_summary(self, cluster_summaries: Dict[int, str], domain_analysis: str) -> str:
        """
        Generates a brief overall summary of the collection.
        """
        try:
            # Take only first 5 cluster summaries
            summaries_text = "\n".join([
                f"Cluster {cluster_id}: {summary}"
                for cluster_id, summary in list(cluster_summaries.items())[:5]
                if cluster_id != -1  # Skip noise points
            ])
            
            prompt = f"""Human: Based on these cluster summaries, provide a brief overview (2-3 sentences) of the main themes:

{summaries_text}

Assistant: Here's the overview:"""
            
            return self._call_llm(prompt, max_tokens=150)
                
        except Exception as e:
            print(f"Error generating overall summary: {e}")
            return "Error: Could not generate overall summary"

    def _get_collections(self) -> List[Dict[str, Any]]:
        """Get list of collections from the API."""
        try:
            response = requests.get(f"{self.backend_url}/api/v1/collections")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching collections: {str(e)}")
            return []
        
    def _get_collection_embeddings(self, collection_name: str) -> Tuple[List[str], np.ndarray]:
        """
        Retrieves document IDs and their embeddings from a collection using VectorStoreService.
        """
        try:
            # Get all documents from the collection
            collection = self.vector_store_service.get_collection(collection_name)
            # Fetch all documents with embeddings, documents, and metadatas fields
            results = list(collection.find({}, {"_id": 1, "embeddings": 1, "documents": 1, "metadatas": 1}))
            if not results:
                print(f"No documents found in collection: {collection_name}")
                return [], np.array([])
            ids = [str(doc["_id"]) for doc in results]
            embeddings = np.array([doc.get("embeddings", []) for doc in results])
            return ids, embeddings
        except Exception as e:
            print(f"Error retrieving embeddings from collection {collection_name}: {e}")
            return [], np.array([])

    def _cluster_embeddings(self, embeddings: np.ndarray, eps: float = 0.5, min_samples: int = 2) -> np.ndarray:
        """
        Clusters embeddings using DBSCAN algorithm.
        """
        try:
            clusterer = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='cosine',
                n_jobs=-1
            )
            return clusterer.fit_predict(embeddings)
        except Exception as e:
            print(f"Error during clustering: {e}")
            return np.array([])
    
    def _get_cluster_documents(self, collection_name: str, doc_ids: List[str], cluster_labels: np.ndarray) -> Dict[int, List[Dict[str, Any]]]:
        """
        Groups documents by their cluster labels.
        """
        try:
            collection = self.vector_store_service.get_collection(collection_name)
            # Convert string IDs to ObjectId if needed
            from bson import ObjectId
            object_ids = [ObjectId(doc_id) for doc_id in doc_ids]
            results = list(collection.find({"_id": {"$in": object_ids}}, {"documents": 1, "metadatas": 1}))
            # Map _id to index for fast lookup
            id_to_result = {str(doc["_id"]): doc for doc in results}
            clusters = {}
            for idx, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                doc_id = doc_ids[idx]
                doc = id_to_result.get(doc_id, {})
                clusters[label].append({
                    'id': doc_id,
                    'content': doc.get('documents', None),
                    'metadata': doc.get('metadatas', None)
                })
            return clusters
        except Exception as e:
            print(f"Error retrieving cluster documents: {e}")
            return {}

    def _get_collection_hash(self, collection_name: str) -> str:
        """Generate a hash of the collection's current state."""
        try:
            collection = self.vector_store_service.get_collection(collection_name)
            # Fetch all metadatas fields
            results = list(collection.find({}, {"metadatas": 1, "_id": 0}))
            # Create a string representation of all document metadata
            metadata_list = [doc.get('metadatas', {}) for doc in results]
            metadata_str = json.dumps(metadata_list, sort_keys=True)
            # Generate hash
            return hashlib.md5(metadata_str.encode()).hexdigest()
        except Exception as e:
            print(f"Error generating collection hash: {e}")
            return ""

    def _get_cache_path(self, collection_name: str) -> Path:
        """Get the path to the cache file for a collection."""
        return self.cache_dir / f"{collection_name}.json"

    def _load_from_cache(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Load analysis results from cache if available and valid."""
        cache_path = self._get_cache_path(collection_name)
        
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
                
            # Verify the collection hasn't changed
            current_hash = self._get_collection_hash(collection_name)
            if cache_data.get('collection_hash') != current_hash:
                print(f"Cache invalid for collection {collection_name} - collection has changed")
                return None
                
            return cache_data
        except Exception as e:
            print(f"Error loading from cache: {e}")
            return None

    def _save_to_cache(self, collection_name: str, analysis_results: Dict[str, Any]) -> None:
        """Save analysis results to cache."""
        try:
            cache_path = self._get_cache_path(collection_name)
            
            # Add collection hash to cache data
            cache_data = {
                **analysis_results,
                'collection_hash': self._get_collection_hash(collection_name),
                'cached_at': time.time()
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving to cache: {e}")

    def _clear_cache(self, collection_name: str) -> None:
        """Clear cache for a collection."""
        try:
            cache_path = self._get_cache_path(collection_name)
            if cache_path.exists():
                cache_path.unlink()
        except Exception as e:
            print(f"Error clearing cache: {e}")

    def analyze_collection(self, collection_name: str, eps: float = 0.5, min_samples: int = 2) -> Dict[str, Any]:
        """
        Main method to analyze a collection:
        1. Checks cache for existing analysis
        2. If cache miss or invalid, performs new analysis:
           - Retrieves embeddings
           - Performs clustering
           - Generates cluster summaries
           - Creates overall summary
        3. Saves results to cache
        """
        print(f"\nAnalyzing collection: {collection_name}")
        
        # Try to load from cache first
        cached_results = self._load_from_cache(collection_name)
        if cached_results:
            print("Using cached analysis results")
            return cached_results
            
        print("Cache miss or invalid - performing new analysis")
        
        # Get embeddings
        print("Retrieving embeddings...")
        doc_ids, embeddings = self._get_collection_embeddings(collection_name)
        
        if len(doc_ids) == 0:
            return {
                "error": f"No documents found in collection: {collection_name}",
                "collection_name": collection_name,
                "total_documents": 0,
                "number_of_clusters": 0,
                "noise_points": 0,
                "cluster_summaries": {},
                "overall_summary": "No documents available for analysis."
            }
        
        # Perform clustering
        print("Performing clustering...")
        cluster_labels = self._cluster_embeddings(embeddings, eps, min_samples)
        
        # Get documents for each cluster
        print("Retrieving cluster documents...")
        clusters = self._get_cluster_documents(collection_name, doc_ids, cluster_labels)
        
        # Analyze domain
        print("Analyzing document domain...")
        all_docs = [doc['content'] for docs in clusters.values() for doc in docs]
        domain_analysis = self._analyze_domain(all_docs)
        print(f"\nDomain Analysis:\n{domain_analysis}\n")
        
        # Generate summaries
        print("Generating cluster summaries...")
        cluster_summaries = {}
        for cluster_id, docs in tqdm(clusters.items(), desc="Summarizing clusters"):
            if cluster_id != -1:  # Skip noise points
                summary = self._summarize_cluster(docs, cluster_id, domain_analysis)
                cluster_summaries[int(cluster_id)] = summary  # Convert numpy.int64 to Python int
        
        # Generate overall summary
        print("Generating overall summary...")
        overall_summary = self._generate_overall_summary(cluster_summaries, domain_analysis)
        
        # Prepare results with converted types
        results = {
            "collection_name": collection_name,
            "total_documents": int(len(doc_ids)),  # Convert numpy.int64 to Python int
            "number_of_clusters": int(len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)),  # Convert numpy.int64 to Python int
            "noise_points": int(sum(1 for label in cluster_labels if label == -1)),  # Convert numpy.int64 to Python int
            "domain_analysis": domain_analysis,
            "cluster_summaries": cluster_summaries,
            "overall_summary": overall_summary
        }
        
        # Save to cache
        self._save_to_cache(collection_name, results)
        
        return results

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze document collections using clustering and LLM summarization")
    parser.add_argument("--collection", type=str, help="Name of the collection to analyze")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps parameter")
    parser.add_argument("--min-samples", type=int, default=2, help="DBSCAN min_samples parameter")
    
    args = parser.parse_args()
    
    service = ClusterAnalysisService()
    
    if not args.collection:
        # List available collections
        collections = service._get_collections()
        print("\nAvailable collections:")
        for collection in collections:
            print(f"- {collection['name']} (Documents: {collection['document_count']})")
        print("\nPlease specify a collection using --collection")
    else:
        # Analyze specified collection
        try:
            results = service.analyze_collection(
                args.collection,
                eps=args.eps,
                min_samples=args.min_samples
            )
            
            # Print results
            print("\nAnalysis Results:")
            print(f"Collection: {results['collection_name']}")
            
            if "error" in results:
                print(f"\nError: {results['error']}")
            else:
                print(f"Total Documents: {results['total_documents']}")
                print(f"Number of Clusters: {results['number_of_clusters']}")
                print(f"Noise Points: {results['noise_points']}")
                
                if results['cluster_summaries']:
                    print("\nCluster Summaries:")
                    for cluster_id, summary in results['cluster_summaries'].items():
                        print(f"\nCluster {cluster_id}:")
                        print(summary)
                    
                    print("\nOverall Summary:")
                    print(results['overall_summary'])
                else:
                    print("\nNo clusters were found in the collection.")
        except Exception as e:
            print(f"\nError analyzing collection: {str(e)}")