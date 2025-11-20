import unittest
from unittest.mock import patch, mock_open
import tempfile
from pathlib import Path
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.package_json_parser import PackageJsonParser

class TestPackageJsonParser(unittest.TestCase):
    def setUp(self):
        self.parser = PackageJsonParser()
        self.sample_package_json = {
             "name": "test-project",
            "version": "1.0.0",
            "dependencies": {
                "express": "^4.18.0",
                "axios": "^1.4.0",
                "lodash": "^4.17.21"
            },
            "devDependencies": {
                "jest": "^29.0.0",
                "eslint": "^8.0.0"
            },
            "peerDependencies": {
                "react": ">=16.8.0"
            }
        }

        self.sample_package_lock = {
            "name": "test-project",
            "version": "1.0.0",
            "lockfileVersion": 3,
            "packages": {
                "": {
                    "name": "test-project",
                    "version": "1.0.0",
                    "dependencies": {
                        "express": "^4.18.0"
                    }
                },
                "node_modules/express": {
                    "version": "4.18.2",
                    "resolved": "https://registry.npmjs.org/express/-/express-4.18.2.tgz",
                    "integrity": "sha512-...",
                    "dependencies": {
                        "body-parser": "1.20.1"
                    }
                },
                "node_modules/axios": {
                    "version": "1.4.0",
                    "resolved": "https://registry.npmjs.org/axios/-/axios-1.4.0.tgz",
                    "integrity": "sha512-..."
                }
            }
        }

    def test_parse_package_json_success(self):
        # create temp file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='json',delete=False) as f:
            json.dump(self.sample_package_json, f)
            temp_path = f.name

        try:
            deps = self.parser.parse_package_json(temp_path)

            self.assertIn('express', deps)
            self.assertIn('axios', deps)
            self.assertIn('lodash', deps)
            self.assertIn('jest', deps)
            self.assertIn('eslint', deps)
            self.assertIn('react', deps)

            self.assertEqual(deps['express'], '^4.18.0')
            self.assertEqual(deps['axios'], '^1.4.0')
            self.assertEqual(deps['lodash'], '^4.17.21')
            self.assertEqual(deps['jest'], '^29.0.0')
            self.assertEqual(deps['eslint'], '^8.0.0')
            self.assertEqual(deps['react'], '>=16.8.0')

        finally:
            os.unlink(temp_path)

    def test_parse_package_lock_success(self):
        # create temp file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='json',delete=False) as f:
            json.dump(self.sample_package_lock, f)
            temp_path = f.name

        try:
            deps = self.parser.parse_package_lock(temp_path)

            self.assertIn('express', deps)
            self.assertIn('axios', deps)

            self.assertEqual(deps['express']['version'], '4.18.2')
            self.assertEqual(deps['axios']['version'], '1.4.0')

        finally:
            os.unlink(temp_path)

      #test package_json where path is invalid
    def test_parse_package_json_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            self.parser.parse_package_json("/nonexistent/path/package.json")
  
    # test package_json where there are no dependencies:
    def test_parse_package_json_no_dependencies(self):
        empty_package_json = {
            "name": "empty-project",
            "version": "1.0.0"
        }

        with tempfile.NamedTemporaryFile(mode='w+', suffix='json',delete=False) as f:
            json.dump(empty_package_json, f)
            temp_path = f.name

        try:
            deps = self.parser.parse_package_json(temp_path)
            self.assertEqual(deps, {})
        finally:
            os.unlink(temp_path)



    # test package_json where json is malformed
    def test_parse_package_json_malformed(self):
        malformed_content = """{
            "name": "malformed-project",
            "version": "1.0.0",
            "dependencies": {
                "express": "^4.18.0",
                "axios": "^1.4.0",
        }"""  # trailing comma and missing closing brace

        with tempfile.NamedTemporaryFile(mode='w+', suffix='json',delete=False) as f:
            f.write(malformed_content)
            temp_path = f.name

        try:
            with self.assertRaises(json.JSONDecodeError):
                self.parser.parse_package_json(temp_path)
        finally:
            os.unlink(temp_path)

    # test package_lock parsing from NPM 7+ format
    def test_parse_package_lock_npm7_format(self):

        with tempfile.NamedTemporaryFile(mode='w+', suffix='json',delete=False) as f:
            json.dump(self.sample_package_lock, f)
            temp_path = f.name
        try:
            packages = self.parser.parse_package_lock(temp_path)
            self.assertIn('express', packages)
            self.assertIn('axios', packages)

            self.assertEqual(packages['express']['version'], '4.18.2')
            self.assertEqual(packages['axios']['version'], '1.4.0')

            self.assertIn('registry.npmjs.org', packages['express']['resolved'])

        finally:
            os.unlink(temp_path)

    # test package_lock where path is invalid
    def test_parse_package_lock_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            self.parser.parse_package_lock("/nonexistent/path/package-lock.json")

    # test to get all dependencies from both package.json and package-lock.json
    def test_get_all_dependencies(self):
        with tempfile.TemporaryDirectory() as temp_dir:

            package_json_path = Path(temp_dir) / "package.json"
            with open(package_json_path, 'w') as f:
                json.dump(self.sample_package_json, f)

            package_lock_path = Path(temp_dir) / "package-lock.json"
            with open(package_lock_path, 'w') as f:
                json.dump(self.sample_package_lock, f)

            parser = PackageJsonParser(temp_dir)
            all_deps = parser.get_all_dependencies()

            self.assertIn('express', all_deps)
            self.assertIn('axios', all_deps)
            self.assertIn('lodash', all_deps)
            self.assertIn('jest', all_deps)
            self.assertIn('eslint', all_deps)
            self.assertIn('react', all_deps)

            self.assertEqual(all_deps['express']['resolved_version'], '4.18.2')  # from lock file
            self.assertEqual(all_deps['axios']['resolved_version'], '1.4.0')     # from lock file
            self.assertEqual(all_deps['lodash']['resolved_version'], '^4.17.21') # from package.json

        
    @patch('builtins.open', new_callable=mock_open, read_data='{}')
    def test_parse_with_mocked_file(self, mock_file):

        with tempfile.TemporaryDirectory() as temp_dir:

            path = Path(temp_dir) / "package.json"

            with patch('pathlib.Path.exists', return_value=True):

                mock_file.return_value.read.return_value = json.dumps(self.sample_package_json)

                deps = self.parser.parse_package_json(path)

                mock_file.assert_called_once_with(Path(path), 'r')

                self.assertEqual(len(deps), 6)  # Should have 6 dependencies total
                self.assertEqual(deps['express'], '^4.18.0')
                self.assertEqual(deps['jest'], '^29.0.0')
                self.assertIn('react', deps)
                
                # Verify the mock's read method was actually called
                mock_file.return_value.read.assert_called_once()


class TestPackageJsonParserEdgeCases(unittest.TestCase):
    def setUp(self):
        self.parser = PackageJsonParser()

    # test parsing monorepo/workspace package.json
    def test_parse_monorepo_package_json(self):
        workspace_package = {
            "name": "monorepo-root",
            "private": True,
            "workspaces": [
                "packages/*",
                "apps/*"
            ],
            "devDependencies": {
                "lerna": "^6.0.0",
                "turbo": "^1.9.0"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(workspace_package, f)
            temp_path = f.name
        
        try:
            deps = self.parser.parse_package_json(temp_path)
            
            # Should still extract devDependencies
            self.assertIn('lerna', deps)
            self.assertIn('turbo', deps)
        finally:
            os.unlink(temp_path)

    # test scoped packages like @types/node and private packages
    def test_scoped_packages(self):
        scoped_package_json = {
            "name": "scoped-project",
            "version": "1.0.0",
            "dependencies": {
                "@types/node": "^18.0.0",
                "@myorg/mypackage": "1.2.3"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w+', suffix='json',delete=False) as f:
            json.dump(scoped_package_json, f)
            temp_path = f.name

        try:
            deps = self.parser.parse_package_json(temp_path)

            self.assertIn('@types/node', deps)
            self.assertIn('@myorg/mypackage', deps)

            self.assertEqual(deps['@types/node'], '^18.0.0')
            self.assertEqual(deps['@myorg/mypackage'], '1.2.3')

        finally:
            os.unlink(temp_path)
    
    #test exotic version identifiers
    def test_exotic_version_identifiers(self):
        exotic_versions = {
            "dependencies": {
                "exact": "1.2.3",
                "caret": "^1.2.3",
                "tilde": "~1.2.3",
                "range": ">=1.2.3 <2.0.0",
                "git": "git+https://github.com/user/repo.git",
                "tag": "latest",
                "file": "file:../local-package",
                "url": "https://github.com/user/repo/tarball/master"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(exotic_versions, f)
            temp_path = f.name
        
        try:
            deps = self.parser.parse_package_json(temp_path)
            
            # All version formats should be preserved
            self.assertEqual(deps['exact'], '1.2.3')
            self.assertEqual(deps['git'], 'git+https://github.com/user/repo.git')
            self.assertEqual(deps['file'], 'file:../local-package')
            self.assertEqual(len(deps), 8)
        finally:
          os.unlink(temp_path)

if __name__ == '__main__':
    unittest.main(verbosity=2)