import unittest
from unittest.mock import Mock, patch


from src.rag.chunker import DocumentChunker

print("Running tests in test_chunker.py")

class TestChunker(unittest.TestCase):

  def setUp(self):
    self.chunker = DocumentChunker(
      chunk_size=100,
      overlap=20,
      min_chunk_size=20
    )

  def test_chunker_initialization(self):

    chunker = DocumentChunker(
      chunk_size=500,
      overlap=50,
      min_chunk_size=100
    )

    self.assertEqual(chunker.chunk_size, 500)
    self.assertEqual(chunker.overlap, 50)
    self.assertEqual(chunker.min_chunk_size, 100)
    self.assertEqual(chunker.chars_per_token, 4)

  def test_empty_text_returns_empty_list(self):
    results = self.chunker.chunk_text("")
    self.assertEqual(results, [])

    results = self.chunker.chunk_text("   ")
    self.assertEqual(results, [])

  def text_is_markdown_heavy_detection(self):
    markdown_text = """
# Main Header

## Section 1

This has **bold text** and [links](http://example.com).

```python
def code():
    pass
```

### Another section
More **bold** and *italic* text.
"""
    plain_text = """
This is just plain text.
It has multiple lines.
But no special markdown formatting.
Just regular sentences.
"""

    self.assertTrue(self.chunker._is_markdown_heavy(markdown_text))
    self.assertFalse(self.chunker._is_markdown_heavy(plain_text))

  def test_chunk_markdown_with_code_blocks(self):
      markdown_text = """# Documentation

This is an intro section.

## Installation

```bash
npm install express
```

## Usage

```javascript
const express = require('express');
const app = express();
```

More text here."""

      chunks = self.chunker._chunk_markdown(markdown_text, {"source": "test.md"})

      code_chunks = [c for c in chunks if c['metadata'].get('chunk_type') == 'code']
      text_chunks = [c for c in chunks if c['metadata'].get('chunk_type') ==  'text']

      self.assertGreater(len(code_chunks), 0)
      self.assertGreater(len(text_chunks), 0)

      bash_chunk = next((c for c in code_chunks if 'npm install' in c['text']), None)
      self.assertIsNotNone(bash_chunk)
      self.assertEqual(bash_chunk['metadata']['language'], 'bash')

  def text_chunk_markdown_header_tracking(self):
    markdown_text = """# Main Title

Content under main title.

## Section A

Content for section A that is long enough to be a chunk.

## Section B 

Content for section B that is also long enough."""
    chunks = self.chunker._chunk_markdown(markdown_text, {})

    section_a_chunk = next((c for c in chunks if 'section a' in c['text'].lower()), None)
    section_b_chunk = next((c for c in chunks if 'section b' in c['text'].lower()), None)

    self.assertIsNotNone(section_a_chunk)
    self.assertIsNotNone(section_b_chunk)

    if section_a_chunk:
       self.assertIn('header', section_a_chunk['metadata'])
    if section_b_chunk:
       self.assertIn('header', section_b_chunk['metadata'])


  def test_chunk_plain_text_paragraph(self):
    plain_text = """This is the first paragraph. It contains some text that should be chunked appropriately based on size.

This is the second paragraph. It also has content that needs to be processed and chunked if it exceeds the chunk size limit.

This is the third paragraph with more content. The chunker should handle multiple paragraphs and combine them when they fit within the chunk size."""

    chunks = self.chunker._chunk_plain_text(plain_text, {})

    self.assertGreater(len(chunks), 0)
    for chunk in chunks:
       self.assertEqual(chunk['metadata'].get('chunk_type'), 'text')
       self.assertGreater(len(chunk['text']), 0)
    
  def test_find_split_point_paragraph(self):
     text = "A" * 350 + "\n\n" + "B" * 350
     split_point = self.chunker._find_split_point(text)
     self.assertEqual(split_point, 350)
     
  def test_find_split_point_sentence_end(self):
     text = "A" * 300 + ". This is a new sentence" + "B" * 200
     split_point = self.chunker._find_split_point(text)
     expected = 300 + len(". ")
     self.assertEqual(split_point, expected)  
  
  def test_find_split_point_space_fallback(self):
     text = "A"*350+ " " + "B"*50
     split_point = self.chunker._find_split_point(text)
     self.assertEqual(split_point, 350)
  
  def test_find_split_point_cutoff(self):
     text = "A"*600
     split_point = self.chunker._find_split_point(text)
     expected_target = self.chunker.chunk_size * self.chunker.chars_per_token
     self.assertEqual(split_point, expected_target)
  
  def test_chunk_code_javascript_functions(self):
     js_code = """export function processData(input) {
    return input.map(x => x * 2);
}

export async function fetchData(url) {
    const response = await fetch(url);
    return response.json();
}

const arrowFunc = async (param) => {
    console.log(param);
    return param;
}

class DataProcessor {
    constructor() {
        this.data = [];
    }
    
    process() {
        return this.data;
    }
}"""
     chunks = self.chunker.chunk_code(js_code, "javascript")
     self.assertGreater(len(chunks), 1)

     function_names=[c['metadata'].get('function_name') for c in chunks]
     print(function_names)
     self.assertIn(any(name == 'processData' in str(name) for name in function_names), True)
     self.assertIn(any(name == 'fetchData' in str(name) for name in function_names), True)
     self.assertIn(any(name == 'arrowFunc' in str(name) for name in function_names), True)

  
  def test_chunk_code_fallback(self):
     python_code = """
def process_data(input_data):
    return [x * 2 for x in input_data]

def fetch_data(url):
    import requests
    return requests.get(url).json()
"""
     chunks = self.chunker.chunk_code(python_code, "python")
     self.assertEqual(len(chunks), 1)
     self.assertEqual(chunks[0]['metadata'].get('chunk_type'), 'code')
     self.assertEqual(chunks[0]['metadata'].get('language'), 'python')
     

  def test_chunk_markdown_long_sections(self):
     long_content = "A" * 500  # Much longer than chunk_size of 100 tokens (400 chars)
     markdown_content = f"""# Header
{long_content}

## Another Header

Short content."""
     
     chunks = self.chunker._chunk_markdown(markdown_content, {})

      # Verify long content is chunked with overlaps -- total content should be a bit longer.
     total_content = "".join([c['text'] for c in chunks])
     self.assertGreater(len(total_content), len(long_content))
  
  def test_metadata_propagation(self):
     text = "Test content for metadata propagation."
     base_metadata = {
            "source": "test_file.md",
            "project": "test_project",
            "custom_field": "custom_value"
        }
     
     chunks = self.chunker.chunk_text(text, base_metadata)
     
     self.assertEqual(len(chunks), 1)
     
     chunk = chunks[0]

     for key, value in base_metadata.items():
        self.assertEqual(chunk['metadata'].get(key), value)
    
     self. assertEqual(chunk['metadata'].get('chunk_type'), 'text')

class TestDocumentChunkerIntegration(unittest.TestCase):
   def setUp(self):
      self.chunker = DocumentChunker(
         chunk_size=150,
         overlap=30,
         min_chunk_size=30
      )

   def test_real_markdown_document(self):
        markdown_doc = """# Express.js Documentation

## Introduction

Express is a minimal and flexible Node.js web application framework that provides a robust set of features for web and mobile applications.

## Installation

You can install Express using npm:

```bash
npm install express --save
```

Or using yarn:

```bash
yarn add express
```

## Basic Example

Here's a simple Express server:

```javascript
const express = require('express');
const app = express();
const port = 3000;

app.get('/', (req, res) => {
    res.send('Hello World!');
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
```

## Routing

Express provides a robust routing mechanism. Routes can have multiple handler functions:

```javascript
app.get('/users/:id', (req, res, next) => {
    // Middleware logic
    next();
}, (req, res) => {
    res.send(`User ${req.params.id}`);
});
```

## Middleware

Middleware functions execute during the request-response cycle. They can execute code, modify the request and response objects, end the cycle, or call the next middleware.

### Built-in Middleware

Express has several built-in middleware functions:

- express.static serves static assets
- express.json parses JSON payloads
- express.urlencoded parses URL-encoded payloads

## Error Handling

Express comes with a built-in error handler. You can also define custom error-handling middleware:

```javascript
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('Something broke!');
});
```

## Conclusion

Express.js is a powerful, unopinionated framework that gives you the flexibility to build web applications and APIs quickly and efficiently.
"""

        chunks = self.chunker._chunk_markdown(markdown_doc, {"source": "express-docs.md"})
        
        # Verify we have both code and text chunks
        code_chunks = [c for c in chunks if c['metadata'].get('chunk_type') == 'code']
        text_chunks = [c for c in chunks if c['metadata'].get('chunk_type') == 'text']
        
        self.assertGreater(len(code_chunks), 0, "Should have code chunks")
        self.assertGreater(len(text_chunks), 0, "Should have text chunks")
        
        # Verify code blocks are properly identified
        js_chunks = [c for c in code_chunks if c['metadata'].get('language') == 'javascript']
        bash_chunks = [c for c in code_chunks if c['metadata'].get('language') == 'bash']
        
        self.assertGreater(len(js_chunks), 0, "Should have JavaScript code chunks")
        self.assertGreater(len(bash_chunks), 0, "Should have Bash code chunks")
        
        # Verify headers are tracked
        headers_found = set()
        for chunk in chunks:
            if 'header' in chunk['metadata']:
                headers_found.add(chunk['metadata']['header'])
        
        self.assertGreater(len(headers_found), 0, "Should track headers in metadata")



if __name__ == "__main__":
    unittest.main(verbosity=2)