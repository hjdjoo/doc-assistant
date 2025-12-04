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


  def scan_project(self) -> Dict:
    self.dependencies = self.parser.get_all_dependencies()

    summary = {
      
    }

  def _detect_framework(self) -> Optional[str]:
    """Detect which framework the project uses"""
    deps = self.dependencies.keys()
    
    if 'react' in deps:
      if 'next' in deps:
        return 'Next.js'
      return 'React'
    elif 'vue' in deps:
      if 'nuxt' in deps:
        return 'Nuxt'
      return 'Vue'
    elif 'angular' in deps:
      return 'Angular'
    elif 'express' in deps:
      return 'Express'
    elif 'fastify' in deps:
      return 'Fastify'
    
    return None