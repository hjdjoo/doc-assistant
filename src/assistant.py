import os
from typing import Dict, List, Optional
from pathlib import Path
from openai import OpenAI


from .fetchers.npm_fetcher import NpmFetcher
from .parsers.package_json_parser import PackageJsonParser
from .rag.chunker import DocumentChunker
from .rag.embedder import Embedder
from .rag.vector_store import VectorStore 

class DocAssistant:
  def __init__(self,
               project_root: str=".",
               llm_base_url: str="http://localhost:1234/v1",
               use_local_llm: bool=True
               ):
    
    self.project_root = Path(project_root)

    self.parser = PackageJsonParser(project_root)
    self.fetcher = NpmFetcher()
    self.vector_store = VectorStore()
    self.chunker = DocumentChunker()
    self.embedder = Embedder()

    if use_local_llm:
      self.llm = OpenAI(
        base_url=llm_base_url,
        api_key="."
      )

    else:
      self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) #not yet implemented, expect error
      self.model = "."

