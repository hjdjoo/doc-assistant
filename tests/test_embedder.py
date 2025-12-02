import unittest
from src.rag.embedder import Embedder

class TestEmbedder(unittest.TestCase):
  def setUp(self):
      self.embedder = Embedder()
  
  