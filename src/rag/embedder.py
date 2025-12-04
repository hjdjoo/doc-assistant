import numpy as np
np.float_ = np.float64
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
import hashlib
import json
from pathlib import Path

class Embedder:
  def __init__(self,
               model_name: str = "all-MiniLM-L6-v2",
               cache_dir: Optional[str] = ".embedding_cache"):
    print(f"Loading embedding model: {model_name}")

    self.model = SentenceTransformer(model_name)
    self.embedding_dim = self.model.get_sentence_embedding_dimension()
    self.cache_dir = Path(cache_dir) if cache_dir else None

    if self.cache_dir:
      self.cache_dir.mkdir(exist_ok=True)
      self.cache_file = self.cache_dir / f"{model_name.replace('/', '_')}_cache.json"
      self.cache = self._load_cache()
    else:
      self.cache = {}


  def _load_cache(self) -> dict:
    if self.cache_file and self.cache_file.exists():
      try:
        with open(self.cache_file, 'r') as f:
          return json.load(f)
      except Exception as e:
        print(f"Failed to load cache: {str(e)}")
        return {}
    return {}
  
  def _save_cache(self):
    if self.cache_file:
      # print("file: ", self.cache_file)
      with open(self.cache_file, 'w') as f:
        json.dump(self.cache, f)
      
  def _get_cache_key(self, text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()
  
  def embed(self, text: Union[str, List[str]], use_cache: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
    is_single = isinstance(text, str)
    texts = [text] if is_single else text
    embeddings = []
    texts_to_embed = []
    cache_indices = []

    for i, t, in enumerate(texts):
      cache_key = self._get_cache_key(t)
      if use_cache and cache_key in self.cache: 
        embeddings.append(np.array(self.cache[cache_key]))
      else:        
        texts_to_embed.append(t)
        cache_indices.append(i)
        embeddings.append(None)  # Placeholder
    
    if texts_to_embed:
      new_embeddings = self.model.encode(
        texts_to_embed,
        convert_to_numpy=True,
        show_progress_bar=len(texts_to_embed) > 10
      )

      if new_embeddings.ndim == 1:
        new_embeddings = [new_embeddings]
      elif new_embeddings.ndim == 2:
        new_embeddings = list(new_embeddings) 

      for idx, emb in zip(cache_indices, new_embeddings):
        embeddings[idx] = np.array(emb)
        # print("embedding: ", emb)
        # print("embeddings: ", embeddings)
        if use_cache:
          cache_key = self._get_cache_key(texts[idx])
          if isinstance(emb, np.ndarray):
            self.cache[cache_key] = emb.tolist()
          # print("cache_key", cache_key)
          else: 
            self.cache[cache_key] = list(emb)

      if use_cache and len(self.cache)%100 == 0:
        self._save_cache()
    
    return embeddings[0] if is_single else embeddings
    
  def embed_query(self, query: str) -> np.ndarray:
    if "multi-qa" in self.model.model_name_or_path:
      query = f"query: {query}"
    return self.embed(query)
  
  def batch_embed(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
      batch = texts[i:i+batch_size]
      embeddings = self.embed(batch)
      all_embeddings.extend(embeddings)
    return all_embeddings
  
  def save_cache(self):
    if self.cache_dir:
      self._save_cache()

  # Compute cosine similarity between two embeddings
  def get_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1_norm, emb2_norm))
  
