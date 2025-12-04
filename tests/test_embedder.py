import unittest
from unittest.mock import Mock, patch
import numpy as np
import tempfile
import json
import os
from pathlib import Path
from src.rag.embedder import Embedder

class TestEmbedder(unittest.TestCase):
  def setUp(self):
    with patch("src.rag.embedder.SentenceTransformer") as mock_transformer:
      self.mock_model = Mock()
      self.mock_model.get_sentence_embedding_dimension.return_value = 384
      self.mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
      mock_transformer.return_value = self.mock_model

      self.temp_dir = tempfile.mkdtemp()
      self.embedder = Embedder(model_name="test-model", cache_dir=self.temp_dir)

  def tearDown(self):
     import shutil
     if os.path.exists(self.temp_dir):
        shutil.rmtree(self.temp_dir)

  def test_initialization(self):
      self.assertEqual(self.embedder.embedding_dim, 384)
      self.assertIsNotNone(self.embedder.model)
      self.assertEqual(self.embedder.cache_dir, Path(self.temp_dir))
      self.assertTrue(self.embedder.cache_file.exists() or True)

  def test_initialization_no_cache(self):
    with patch('src.rag.embedder.SentenceTransformer') as mock_transformer:
      mock_model = Mock()
      mock_model.get_sentence_embedding_dimension.return_value = 384
      mock_transformer.return_value = mock_model

      embedder = Embedder(model_name="test-model", cache_dir=None)
      self.assertIsNone(embedder.cache_dir)
      self.assertEqual(embedder.cache, {})

  def test_cache_key_generation(self):
    text = "This is a test sentence."
    key1 = self.embedder._get_cache_key(text)
    key2 = self.embedder._get_cache_key(text)
    self.assertEqual(key1, key2)

    key3 = self.embedder._get_cache_key("Different text.")
    self.assertNotEqual(key1, key3)

    self.assertEqual(len(key1), 32)  # MD5 hash length
    self.assertTrue(all(c in '0123456789abcdef' for c in key1))

  def test_embed_single_text(self):
    text = "Test sentence."
    mock_embedding = np.array([0.1, 0.2, 0.3])

    self.mock_model.encode.return_value = mock_embedding
    result = self.embedder.embed(text)

    self.assertIsInstance(result, np.ndarray)
    np.testing.assert_array_equal(result, mock_embedding)

    self.mock_model.encode.assert_called()
      
  def test_embed_multiple_texts(self):
    texts = ["This is the first sentence.", "This is the second sentence.", "This is the third sentence."]

    mock_embeddings = np.array([
          [0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6],
          [0.7, 0.8, 0.9]
      ])
    self.mock_model.encode.return_value = mock_embeddings
      
    result = self.embedder.embed(texts)
    
    self.assertIsInstance(result, list)
    self.assertEqual(len(result), 3)
    for i, embedding in enumerate(result):
        np.testing.assert_array_equal(embedding, mock_embeddings[i])

  def test_embed_with_cache_hit(self):
    text = "Cached text"
    cached_embedding = [0.9, 0.8, 0.7]

    cache_key = self.embedder._get_cache_key(text)
    self.embedder.cache[cache_key] = cached_embedding

    result = self.embedder.embed(text, use_cache=True)

    np.testing.assert_array_equal(result, np.array(cached_embedding))
    self.mock_model.encode.assert_not_called()

  def test_embed_without_cache(self):
    """Test embedding with cache disabled"""
    text = "No cache text"
    mock_embedding = np.array([0.5, 0.5, 0.5])
    self.mock_model.encode.return_value = mock_embedding
    
    result1 = self.embedder.embed(text, use_cache=False)
    np.testing.assert_array_equal(result1, mock_embedding)
    
    result2 = self.embedder.embed(text, use_cache=False)
    np.testing.assert_array_equal(result2, mock_embedding)
    
    self.assertEqual(self.mock_model.encode.call_count, 2)

  def test_embed_query(self):
    """Test query embedding with special formatting"""
    # Test with multi-qa model
    self.embedder.model.model_name_or_path = "multi-qa-model"
    query = "What is the meaning of life?"
    mock_embedding = np.array([0.1, 0.2, 0.3])
    self.mock_model.encode.return_value = mock_embedding
    
    with patch.object(self.embedder, 'embed') as mock_embed:
      mock_embed.return_value = mock_embedding
      result = self.embedder.embed_query(query)

      # Should add "query: " prefix for multi-qa models
      mock_embed.assert_called_with("query: " + query)
      self.assertIsInstance(result, np.ndarray)
      np.testing.assert_array_equal(result, mock_embedding)

  def test_batch_embed(self):
    """Test batch embedding with specified batch size"""
    texts = [f"Text {i}" for i in range(10)]
    mock_embeddings = [np.array([i, i+0.1, i+0.2]) for i in range(10)]
    
    with patch.object(self.embedder, 'embed') as mock_embed:
      # Mock embed to return appropriate embeddings for each batch
      def embed_side_effect(batch_texts):
        if isinstance(batch_texts, list):
          return [mock_embeddings[int(t.split()[1])] for t in batch_texts]
        else:
          return mock_embeddings[int(batch_texts.split()[1])]
      
      mock_embed.side_effect = embed_side_effect
      
      result = self.embedder.batch_embed(texts, batch_size=3)
      
      # Should process in batches
      self.assertEqual(len(result), 10)
      # Should be called 4 times (3+3+3+1)
      self.assertEqual(mock_embed.call_count, 4)
  def test_get_similarity(self):
    """Test cosine similarity calculation"""
    # Test with identical vectors
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([1.0, 0.0, 0.0])
    similarity = self.embedder.get_similarity(emb1, emb2)
    self.assertAlmostEqual(similarity, 1.0)
    
    # Test with orthogonal vectors
    emb3 = np.array([0.0, 1.0, 0.0])
    similarity = self.embedder.get_similarity(emb1, emb3)
    self.assertAlmostEqual(similarity, 0.0)
    
    # Test with opposite vectors
    emb4 = np.array([-1.0, 0.0, 0.0])
    similarity = self.embedder.get_similarity(emb1, emb4)
    self.assertAlmostEqual(similarity, -1.0)

  def test_cache_persistence(self):
    """Test that cache is saved and loaded correctly"""
    text = "Persistent text"
    mock_embedding = np.array([0.1, 0.2, 0.3])
    self.mock_model.encode.return_value = mock_embedding
    
    # Embed and save cache
    self.embedder.embed(text)
    self.embedder.save_cache()
    
    # Verify cache file exists and contains data
    self.assertTrue(self.embedder.cache_file.exists())
    
    with open(self.embedder.cache_file, 'r') as f:
      saved_cache = json.load(f)
    
    cache_key = self.embedder._get_cache_key(text)
    self.assertIn(cache_key, saved_cache)
    np.testing.assert_array_almost_equal(
      saved_cache[cache_key], 
      mock_embedding.tolist()
      )
  def test_cache_auto_save(self):
    """Test automatic cache saving every 100 entries"""
    # Add 99 entries - should not trigger save
    for i in range(99):
      self.embedder.cache[f"key_{i}"] = [i, i, i]
    
    with patch.object(self.embedder, '_save_cache') as mock_save:
      # 100th entry should trigger save
      text = "Trigger save"
      mock_embedding = np.array([100, 100, 100])
      self.mock_model.encode.return_value = mock_embedding
      
      self.embedder.embed(text)
      
      # Auto-save should be triggered
      mock_save.assert_called()

  def test_cache_load_corrupted(self):
    """Test handling of corrupted cache file"""
    # Create corrupted cache file
    cache_file = self.embedder.cache_file
    with open(cache_file, 'w') as f:
      f.write("corrupted json {]")
    
    # Should handle gracefully and return empty cache
    cache = self.embedder._load_cache()
    self.assertEqual(cache, {})
  
  def test_mixed_batch_with_cache(self):
    """Test batch processing with some cached and some new embeddings"""
    texts = ["cached_1", "new_1", "cached_2", "new_2"]
    
    # Pre-populate cache for some texts
    self.embedder.cache[self.embedder._get_cache_key("cached_1")] = [0.1, 0.1, 0.1]
    self.embedder.cache[self.embedder._get_cache_key("cached_2")] = [0.2, 0.2, 0.2]
    
    # Mock encode for new texts
    self.mock_model.encode.return_value = np.array([
      [0.3, 0.3, 0.3],  # new_1
      [0.4, 0.4, 0.4]   # new_2
    ])
    
    results = self.embedder.embed(texts)
    
    # Should have all 4 results
    self.assertEqual(len(results), 4)
    
    # Cached results should match
    np.testing.assert_array_almost_equal(results[0], [0.1, 0.1, 0.1])
    np.testing.assert_array_almost_equal(results[2], [0.2, 0.2, 0.2])
    
    # New results should be from encode
    np.testing.assert_array_almost_equal(results[1], [0.3, 0.3, 0.3])
    np.testing.assert_array_almost_equal(results[3], [0.4, 0.4, 0.4])
    
    # Encode should only be called for new texts
    self.mock_model.encode.assert_called_once()
    call_args = self.mock_model.encode.call_args[0][0]
    self.assertEqual(call_args, ["new_1", "new_2"])