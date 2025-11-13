import json
import requests
from pathlib import Path
import ollama

def parse_package_json (file_path="package.json"):
  # Parse package.json for dependencies
  with open(file_path, 'r') as f:
    data = json.load(file_path)
  deps = {}
  if 'dependencies' in data:
    deps.update(data['dependencies'])
  if 'dev_depdendencies' in data:
    deps.update(data['dev_dependencies'])
  return deps

