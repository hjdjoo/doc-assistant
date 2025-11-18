import unittest
import sys
from pathlib import Path

# add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_all_tests():
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()

def run_specific_module(module_name: str):
    loader = unittest.TestLoader()
    try:
      # check module name: "parser", "fetcher", or "rag" and import accordingly
      if module_name == "parser":
          from tests import test_parser
          suite = loader.loadTestsFromModule(test_parser)
      elif module_name == "fetcher":
          from tests import test_npm_fetch
          suite = loader.loadTestsFromModule(test_npm_fetch)
      elif module_name == "rag":
          from tests import test_rag
          suite = loader.loadTestsFromModule(test_rag)
      else:
          print(f"Unknown module name: {module_name}")
          return False
      
      runner = unittest.TextTestRunner(verbosity=2)
      result = runner.run(suite)
      return result.wasSuccessful()
    
    except ImportError as e:
      print(f"Failed to import module for testing: {str(e)}")
      return False
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        module = sys.argv[1]
        success = run_specific_module(module)
    else:
        success = run_all_tests()

    sys.exit(0 if success else 1)