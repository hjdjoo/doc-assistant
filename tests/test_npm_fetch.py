import unittest
from unittest.mock import patch, Mock
import json
import requests
import os
import sys
import time


from src.fetchers.npm_fetcher import NpmFetcher


class TestNpmFetcher(unittest.TestCase):

    def setUp(self):
        self.fetcher = NpmFetcher()
        self.sample_npm_response = {
            "name": "express",
            "description": "Fast, unopinionated, minimalist web framework",
            "dist-tags": {
                "latest": "4.18.2",
                "next": "5.0.0-beta.1"
            },
            "versions": {
                "4.18.2": {
                    "name": "express",
                    "version": "4.18.2",
                    "description": "Fast, unopinionated, minimalist web framework",
                    "dependencies": {
                        "body-parser": "1.20.1",
                        "cookie": "0.5.0"
                    }
                },
                "4.18.0": {
                    "name": "express",
                    "version": "4.18.0",
                    "dependencies": {}
                }
            },
            "readme": "# Express\n\nFast, unopinionated, minimalist web framework for Node.js",
            "homepage": "http://expressjs.com/",
            "repository": {
                "type": "git",
                "url": "git://github.com/expressjs/express.git"
            },
            "keywords": ["express", "framework", "web"],
            "license": "MIT",
            "time": {
                "4.18.2": "2022-10-08T17:18:31.453Z"
            }
        }
    
    # mock and test "get" requests
    @patch('requests.Session.get')
    def test_fetch_package_info_success(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_npm_response
        mock_get.return_value = mock_response

        result = self.fetcher.fetch_package_info("express", "4.18.2")

        self.assertEqual(result['name'], "express")
        self.assertEqual(result['version'], "4.18.2")
        self.assertEqual(result['description'], "Fast, unopinionated, minimalist web framework")
        self.assertEqual(result['homepage'], "http://expressjs.com/")
        self.assertEqual(result['repository']['url'], "git://github.com/expressjs/express.git")
        self.assertEqual(result['keywords'], ["express", "framework", "web"])
        self.assertEqual(result['license'], "MIT")
        self.assertEqual(result['time_updated'], "2022-10-08T17:18:31.453Z")
        self.assertIn('body-parser', result['dependencies'])
        self.assertIn('cookie', result['dependencies'])

    # get latest version if none specified
    @patch('requests.Session.get')
    def test_fetch_package_info_latest(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_npm_response
        mock_get.return_value = mock_response

        result = self.fetcher.fetch_package_info("express")

        self.assertEqual(result['version'], '4.18.2')

    # test when requested version does not exist
    @patch('requests.Session.get')
    def test_fetch_package_info_version_not_found(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_npm_response
        mock_get.return_value = mock_response

        result = self.fetcher.fetch_package_info("express", "99.99.99")

        self.assertEqual(result['name'], "express")
        self.assertIsNone(result['version'])
        self.assertEqual(result['description'], "Fast, unopinionated, minimalist web framework")

    # test network error handling
    @patch('requests.Session.get')
    def test_fetch_package_info_network_error(self, mock_get):
        mock_get.side_effect = requests.RequestException("Network error")

        result = self.fetcher.fetch_package_info("express", "4.18.2")

        self.assertEqual(result['name'], "express")
        self.assertIn("Failed to fetch package info", result['error'])

    # test multiple packages with rate limiting
    @patch('requests.Session.get')
    def test_fetch_multiple_packages(self, mock_get):
        
        responses = []

        for pkg_name in ["express", "lodash", "axios"]:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                **self.sample_npm_response,
                "name": pkg_name
            }
            responses.append(mock_response)

        mock_get.side_effect = responses

        packages = {
            "express": "4.18.2",
            "lodash": "4.17.21",
            "axios": "1.4.0"
        }

        start_time = time.time()
        results = self.fetcher.fetch_multiple_packages(packages, delay=0.1)
        elapsed_time = time.time() - start_time

        self.assertEqual(len(results), 3)
        self.assertIn("express", results)
        self.assertIn("lodash", results)
        self.assertIn("axios", results)
        self.assertGreaterEqual(elapsed_time, 0.2) 

    #test that version normalization works
    @patch('requests.Session.get')
    def test_version_normalization(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_npm_response
        mock_get.return_value = mock_response

        test_cases = [
            ('^4.18.0', '4.18.0'),
            ('~4.18.0', '4.18.0'),
            ('>=4.18.0', '4.18.0'),
            ('4.18.0', '4.18.0'),
            ('^4.18.0 <5.0.0', '4.18.0 <5.0.0')
        ]

        for input_version, expected_version in test_cases:
            with self.subTest(input_version=input_version, expected_version=expected_version):
                result = self.fetcher.fetch_package_info("express", input_version)
                self.assertIsNotNone(result)

    # test that get_package_types works
    @patch('requests.Session.get')
    def test_get_package_types(self, mock_get):
        mock_response_success=Mock()
        mock_response_success.status_code=200
        mock_get.return_value=mock_response_success

        types_package = self.fetcher.get_package_types('node')
        self.assertEqual(types_package, '@types/node')

        # test package with no types
        mock_response_fail = Mock()
        mock_response_fail.status_code = 404
        mock_get.return_value = mock_response_fail
        
        types_package = self.fetcher.get_package_types('express')
        self.assertIsNone(types_package)

    # test get package types for scoped packages
    @patch('requests.Session.get')
    def test_get_package_types_scoped(self, mock_get):
        mock_response_success=Mock()
        mock_response_success.status_code=200
        mock_get.return_value=mock_response_success

        types_package = self.fetcher.get_package_types('@angular/core')
        self.assertEqual(types_package, '@types/angular__core')

    
