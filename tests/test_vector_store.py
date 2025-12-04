import unittest
from unittest.mock import MagicMock, Mock, patch
from src.rag.vector_store import VectorStore
from src.rag.embedder import Embedder
import numpy as np


class TestVectorStore(unittest.TestCase):

  @patch('src.rag.vector_store.chromadb.PersistentClient')
  def setUp(self, mock_client):
    self.mock_collection = Mock()
    self.mock_collection.name = "test_collection"

    self.mock_client_instance = Mock()
    self.mock_client_instance.get_collection.return_value = self.mock_collection
    self.mock_client_instance.create_collection.return_value = self.mock_collection
    mock_client.return_value = self.mock_client_instance

    self.vector_store = VectorStore(
        collection_name="test_collection",
        persist_directory="./test_chroma_db"
    )

  def test_initialization_with_existing_collection(self):
    self.mock_client_instance.get_collection.assert_called_with(name="test_collection")
    self.assertEqual(self.vector_store.collection_name, "test_collection")

  @patch('src.rag.vector_store.chromadb.PersistentClient')
  def test_initialization_with_new_collection(self, mock_client):

    mock_client_instance = Mock()
    mock_client_instance.get_collection.side_effect = Exception("Collection not found")
    
    mock_collection = Mock()
    mock_client_instance.create_collection.return_value = mock_collection

    mock_client.return_value = mock_client_instance

    VectorStore(
        collection_name="new_collection"
        )
    
    mock_client_instance.create_collection.assert_called_with(
      name = "new_collection",
      metadata={"hnsw:space": "cosine"}
      )
    
  def test_add_single_document(self):
    text = "This is a test document."
    embedding = np.array([0.1, 0.2, 0.3])

    metadata = {"source": "test.txt", "page": 1}
    doc_id = "test_id_1"

    result = self.vector_store.add(
      text,
      embedding,
      metadata,
      doc_id
    )

    self.assertEqual(result, [doc_id])
    self.mock_collection.add.assert_called_once()
    call_args = self.mock_collection.add.call_args.kwargs
    
    self.assertEqual(call_args['documents'], [text])
    self.assertEqual(call_args['embeddings'], [[0.1, 0.2, 0.3]])
    self.assertEqual(call_args['ids'], [doc_id])
    
    self.assertIn('indexed_at', call_args['metadatas'][0])
    self.assertEqual(call_args['metadatas'][0]['source'], 'test.txt')
    self.assertEqual(call_args['metadatas'][0]['page'], 1)

  def test_add_multiple_documents(self):
    """Test adding multiple documents at once"""
    texts = ["Doc 1", "Doc 2", "Doc 3"]
    embeddings = [
        np.array([0.1, 0.1, 0.1]),
        np.array([0.2, 0.2, 0.2]),
        np.array([0.3, 0.3, 0.3])
    ]
    metadatas = [
        {"type": "A"},
        {"type": "B"},
        {"type": "C"}
    ]
    
    result = self.vector_store.add(texts, embeddings, metadatas)
    
    # Should return list of generated IDs
    self.assertEqual(len(result), 3)
    
    # Check collection.add was called correctly
    call_args = self.mock_collection.add.call_args.kwargs
    self.assertEqual(call_args['documents'], texts)
    self.assertEqual(len(call_args['embeddings']), 3)
    self.assertEqual(len(call_args['metadatas']), 3)

  def test_add_with_auto_generated_ids(self):
    """Test that IDs are auto-generated when not provided"""
    with patch('src.rag.vector_store.uuid.uuid4') as mock_uuid:
      mock_uuid.side_effect = [
          Mock(return_value="uuid1"),
          Mock(return_value="uuid2")
      ]
      
      texts = ["Doc 1", "Doc 2"]
      embeddings = [np.array([0.1]), np.array([0.2])]
      
      result = self.vector_store.add(texts, embeddings)
      self.assertEqual(len(result), 2)
      # Should generate UUIDs
      self.assertEqual(mock_uuid.call_count, 2)

  def test_add_without_metadata(self):
    """Test adding documents without metadata"""
    text = "No metadata doc"
    embedding = np.array([0.5, 0.5])
    
    self.vector_store.add(text, embedding)
    
    # Should still add indexed_at timestamp
    call_args = self.mock_collection.add.call_args.kwargs
    self.assertIn('indexed_at', call_args['metadatas'][0])


  def test_search_with_query_text(self):
    """Test searching with query text"""
    self.mock_collection.query.return_value = {
        'ids': [['id1', 'id2']],
        'documents': [['Document 1', 'Document 2']],
        'metadatas': [[{'source': 'file1'}, {'source': 'file2'}]],
        'distances': [[0.1, 0.3]]
    }
    
    results = self.vector_store.search(query="test query", top_k=2)
    
    # Check query was called with correct parameters
    self.mock_collection.query.assert_called_with(
        query_texts=["test query"],
        n_results=2
    )
    
    # Check formatted results
    self.assertEqual(len(results), 2)
    self.assertEqual(results[0]['id'], 'id1')
    self.assertEqual(results[0]['text'], 'Document 1')
    self.assertEqual(results[0]['metadata']['source'], 'file1')
    self.assertAlmostEqual(results[0]['score'], 0.9)  # 1 - 0.1
    
    self.assertEqual(results[1]['id'], 'id2')
    self.assertAlmostEqual(results[1]['score'], 0.7)  # 1 - 0.3

  def test_search_with_query_embedding(self):
    """Test searching with query embedding"""
    query_embedding = np.array([0.1, 0.2, 0.3])
    
    # Mock the query embedding to have the expected structure
    mock_query_emb = Mock()
    mock_query_emb.embedding = query_embedding
    
    self.mock_collection.query.return_value = {
      'ids': [['id1']],
      'documents': [['Document 1']],
      'metadatas': [[{'source': 'file1'}]],
      'distances': [[0.2]]
    }
    
    self.vector_store.search(None, query_embedding=mock_query_emb, top_k=1)
    
    # Check query was called with embeddings
    call_args = self.mock_collection.query.call_args.kwargs
    self.assertIn('query_embeddings', call_args)
    self.assertEqual(call_args['n_results'], 1)

  def test_search_with_metadata_filter(self):
    """Test searching with metadata filters"""
    self.mock_collection.query.return_value = {
      'ids': [['id1']],
      'documents': [['Filtered doc']],
      'metadatas': [[{'type': 'report', 'year': 2023}]],
      'distances': [[0.15]]
    }
    
    filter_metadata = {
        'type': 'report',
        'year': 2023
    }
    
    self.vector_store.search(
      query="test", 
      filter_metadata=filter_metadata,
      top_k=5
    )
    
    # Check where clause was constructed correctly
    call_args = self.mock_collection.query.call_args.kwargs
    self.assertIn('where', call_args)
    self.assertEqual(call_args['where']['type'], {'$eq': 'report'})
    self.assertEqual(call_args['where']['year'], {'$eq': 2023})

  def test_search_with_list_filter(self):
    """Test searching with list values in metadata filter"""
    filter_metadata = {
      'category': ['tech', 'science', 'engineering']
    }
    
    self.mock_collection.query.return_value = {
      'ids': [[]],
      'documents': [[]],
      'metadatas': [[]],
      'distances': [[]]
    }
    
    self.vector_store.search(query="test", filter_metadata=filter_metadata)
    
    # Check $in operator was used for list
    call_args = self.mock_collection.query.call_args.kwargs
    self.assertEqual(
      call_args['where']['category'],
      {'$in': ['tech', 'science', 'engineering']}
    )
  
  def test_search_no_query_error(self):
    """Test that search raises error when no query is provided"""
    with self.assertRaises(ValueError) as context:
      self.vector_store.search()
    
    self.assertIn("Provide query or query embedding", str(context.exception))

  def test_update_documents(self):
    """Test updating documents"""
    ids = ['id1', 'id2']
    texts = ['Updated text 1', 'Updated text 2']
    embeddings = [np.array([0.1]), np.array([0.2])]
    metadatas = [{'version': 2}, {'version': 2}]
    
    self.vector_store.update(ids, texts, embeddings, metadatas)
    
    # Check update was called correctly
    call_args = self.mock_collection.update.call_args.kwargs
    self.assertEqual(call_args['ids'], ids)
    self.assertEqual(call_args['documents'], texts)
    self.assertEqual(len(call_args['embeddings']), 2)
    
    # Check updated_at timestamp was added
    for meta in call_args['metadatas']:
      self.assertIn('updated_at', meta)
      self.assertEqual(meta['version'], 2)

  def test_update_single_document(self):
    """Test updating a single document"""
    self.vector_store.update(
      ids='single_id',
      texts='Updated text',
      embeddings=np.array([0.5, 0.5]),
      metadatas={'status': 'revised'}
    )
    
    # Should convert to lists
    call_args = self.mock_collection.update.call_args.kwargs
    self.assertEqual(call_args['ids'], ['single_id'])
    self.assertEqual(call_args['documents'], ['Updated text'])
    self.assertEqual(len(call_args['embeddings']), 1)
    self.assertEqual(len(call_args['metadatas']), 1)

  def test_update_partial_fields(self):
    """Test updating only some fields"""
    # Update only metadata
    self.vector_store.update(
        ids=['id1'],
        metadatas=[{'new_field': 'value'}]
    )
    
    call_args = self.mock_collection.update.call_args.kwargs
    self.assertIn('ids', call_args)
    self.assertIn('metadatas', call_args)
    self.assertNotIn('documents', call_args)
    self.assertNotIn('embeddings', call_args)

  def test_get_by_package(self):
    """Test retrieving documents by package name"""
    self.mock_collection.get.return_value = {
      'ids': ['id1', 'id2'],
      'documents': ['Doc about express', 'Another express doc'],
      'metadatas': [
        {'package': 'express', 'version': '4.18.0'},
        {'package': 'express', 'version': '4.18.2'}
      ]
    }
    
    results = self.vector_store.get_by_package('express')
    
    # Check get was called with correct where clause
    self.mock_collection.get.assert_called_with(
      where={'package': {'$eq': 'express'}}
    )
    
    # Check formatted results
    self.assertEqual(len(results), 2)
    self.assertEqual(results[0]['id'], 'id1')
    self.assertEqual(results[0]['text'], 'Doc about express')
    self.assertEqual(results[0]['metadata']['package'], 'express')

  def test_get_by_package_empty_results(self):
    """Test get_by_package with no results"""
    self.mock_collection.get.return_value = {
        'ids': [],
        'documents': [],
        'metadatas': []
    }
    
    results = self.vector_store.get_by_package('nonexistent')
    
    self.assertEqual(results, [])

  def test_clear_collection(self):
    """Test clearing the collection"""
    self.vector_store.clear_collection()
    
    # Should delete and recreate collection
    self.mock_client_instance.delete_collection.assert_called_with("test_collection")
    self.mock_client_instance.create_collection.assert_called_with(
        name=self.mock_collection.name,
        metadata={"hnsw:space": "cosine"}
    )

  def test_get_stats(self):
    """Test getting collection statistics"""
    self.mock_collection.count.return_value = 42
    self.mock_collection.get.return_value = {
        'metadatas': [{'field1': 'value', 'field2': 'value', 'field3': 'value'}]
    }
    
    stats = self.vector_store.get_stats()
    
    self.assertEqual(stats['total_documents'], 42)
    self.assertEqual(stats['collection_name'], 'test_collection')
    self.assertEqual(set(stats['metadata_fields']), {'field1', 'field2', 'field3'})

  def test_get_stats_empty_collection(self):
    """Test getting stats for empty collection"""
    self.mock_collection.count.return_value = 0
    self.mock_collection.get.return_value = {
        'metadatas': []
    }
    
    stats = self.vector_store.get_stats()
    
    self.assertEqual(stats['total_documents'], 0)
    self.assertEqual(stats['metadata_fields'], [])


class TestVectorStoreEdgeCases(unittest.TestCase):
  """Test edge cases and error handling"""
  
  @patch('src.rag.vector_store.chromadb.PersistentClient')
  def setUp(self, mock_client):
    self.mock_collection = Mock()
    self.mock_client_instance = Mock()
    self.mock_client_instance.get_collection.return_value = self.mock_collection
    mock_client.return_value = self.mock_client_instance
    
    self.vector_store = VectorStore()

  def test_handle_numpy_types(self):
    """Test that numpy arrays are properly converted"""
    # Test with different numpy dtypes
    embedding_float32 = np.array([0.1, 0.2], dtype=np.float32)
    embedding_float64 = np.array([0.3, 0.4], dtype=np.float64)
    
    self.vector_store.add(
        ["Text 1", "Text 2"],
        [embedding_float32, embedding_float64]
    )
    
    call_args = self.mock_collection.add.call_args.kwargs
    
    # All embeddings should be converted to lists
    for emb in call_args['embeddings']:
        self.assertIsInstance(emb, list)

  def test_empty_search_results(self):
    """Test handling of empty search results"""
    self.mock_collection.query.return_value = {
        'ids': [[]],
        'documents': [[]],
        'metadatas': [[]],
        'distances': [[]]
    }
    
    results = self.vector_store.search(query="no results query")
    
    self.assertEqual(results, [])

  def test_missing_metadata_in_results(self):
    """Test handling when metadata is missing in results"""
    self.mock_collection.query.return_value = {
        'ids': [['id1']],
        'documents': [['Document']],
        'metadatas': None,  # No metadata
        'distances': [[0.5]]
    }
    
    results = self.vector_store.search(query="test")
    
    # Should handle missing metadata gracefully
    self.assertEqual(len(results), 1)
    self.assertEqual(results[0]['metadata'], {})


if __name__ == '__main__':
    unittest.main(verbosity=2)