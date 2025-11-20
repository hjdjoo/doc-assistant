import chromadb
from chromadb.config import Settings
from typing import List, Union, Dict, Any, Optional
import numpy as np
import uuid
from datetime import datetime

np.float_ = np.float64

class VectorStore:
  def __init__(
      self,
      collection_name: str = "npm_docs",
      persist_directory: str = "./chroma_db"
  ):
    self.client = chromadb.PersistentClient(
      path = persist_directory,
      settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )

    try:
      self.collection = self.client.get_collection(name=collection_name)
      print(f"Using existing collection: {collection_name}")
    except:
      self.collection = self.client.create_collection(
        name=collection_name, 
        metadata={"hnsw:space": "cosine"}
        )
      print(f"Created new collection: {collection_name}")

    self.collection_name = collection_name

  def add(self,
          text: Union[str, List[str]],
          embedding: Union[np.ndarray, List[np.ndarray]],
          metadata: Optional[Union[Dict, List[Dict]]] = None,
          ids: Optional[Union[str, List[str]]] = None
          ) -> List[str]:
    
    if isinstance(text, str):
      texts = [text]
      embeddings = [embedding]
      metadatas = [metadata] if metadata else [{}]
      ids = [ids] if ids else None

    else:
      texts = text
      embeddings = embedding
      metadatas = metadata if metadata else [{} for _ in texts]

    if not ids:
      ids = [str(uuid.uuid4()) for _ in texts]

    for meta in metadatas:
      meta['indexed_at'] = datetime.now().isoformat()
    
    embeddings_list = [
      emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings
    ]
    
    self.collection.add(
      documents = texts,
      embeddings = embeddings_list,
      metadatas = metadatas,
      ids = ids
    )

    return ids
  
  def search(self,
             query: str = None,
             query_embedding: np.ndarray = None,
             filter_metadata: Dict = None,
             top_k: int = 5) -> List[Dict]:
    if query is None and query_embedding is None:
      raise ValueError("Provide query or query embedding")
    
    query_params = {
      'n_results': top_k
    }

    if query:
      query_params['query_texts'] = [query]
    else:
      query_params['query_embeddings'] = [query.embedding.tolist()]

    if filter_metadata:
      where_clause = {}
      for key, value in filter_metadata.items():
        if isinstance(value, list):
          where_clause[key] = {"$in": value}
        else:
          where_clause[key] = {"$eq": value}

      query_params['where'] = where_clause

    results = self.collection.query(**query_params)

    formatted_results = []
    for i in range(len(results['ids'][0])):
      formatted_results.append({
        'id': results['ids'][0][i],
        'text': results['documents'][0][i],
        'metadata': results['metadata'][0][i] if results['metadata'] else {},
        'score': 1 - results['distances'][0][i]
      })

    return formatted_results
  
  def update(self,
             ids: Union[str, List[str]],
             texts: Optional[Union[str, List[str]]] = None,
             embeddings: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
             metadatas: Optional[Union[Dict, List[Dict]]] = None
             ):
    update_params = {
      'ids': ids if isinstance(ids, list) else [ids]
    }

    if texts:
      update_params['documents'] = texts if isinstance(texts, list) else [texts]
    
    if embeddings is not None:
      if isinstance(embeddings, list):
        update_params['embeddings'] = [
          e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings
        ]
      else:
        update_params['embeddings'] = [
          embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        ]
    
    if metadatas:
      update_params['metadatas'] = metadatas if isinstance(metadatas, list) else [metadatas]
      for meta in update_params['metadatas']:
        meta['updated_at'] = datetime.now().isoformat()

    self.collection.update(**update_params)

def delete(self, ids: Union[str, List[str]]):
  self.collection.delete(ids = ids if isinstance(ids, list) else [ids])

def get_by_package(self, package_name: str) -> List[Dict]:
  results = self.collection.get(where= {
    "package": { 
      "$eq": package_name
      }
    })
  
  formatted = []

  if results['ids']:
    for i in range(len(results['ids'])):
      formatted.append({
        'id': results['ids'][i],
        'text': results['documents'][i] if results['documents'] else None,
        'metadata': results['metadatas'][i] if results['metadatas'] else {}
      })

  return formatted

def clear_collection(self):
  self.client.delete_collection(self.collection_name)
  self.collection = self.client.create_collection(
    name = self.collection.name,
    metadata = {
      "hnsw:space": "cosine"
    }
  )
  print(f"Cleared Collection: {self.collection_name}")

def get_stats(self) -> Dict:
  count = self.collection.count()

  sample = self.collection.get(limit=1)
  metadata_fields = list(sample['metadatas'][0].keys()) if sample['metadatas'] else []

  return {
    'total_documents': count,
    'collection_name': self.collection_name,
    'metadata_fields': metadata_fields
  }