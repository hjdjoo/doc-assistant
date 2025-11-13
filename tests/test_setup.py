import sys

print(f"python {sys.version.split()[0]} installed")

try:
  import ollama
  print("✅ Ollama library installed")
except ImportError:
  print("❌ Ollama library missing")

try:
    import chromadb
    print("✅ ChromaDB installed")
except ImportError:
    print("❌ ChromaDB missing")

try:
    import sentence_transformers
    print("✅ Sentence Transformers installed")
except ImportError:
    print("❌ Sentence Transformers missing")

    
try:
    import ollama
    response = ollama.list()
    print(f"✅ Ollama running with {len(response['models'])} model(s)")
except:
    print("❌ Ollama not running - run 'ollama serve' in another terminal")