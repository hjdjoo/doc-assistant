import requests
import json
from typing import Dict, Optional, List
from datetime import datetime
import time

class NpmFetcher:

    def __init__(self, cache_dir: Optional [str] = None):
        self.npm_registry_url = "https://registry.npmjs.org/"
        self.session = requests.Session()
        self.cache_dir = cache_dir # Not implemented yet

    def fetch_package_info(self, package_name: str, version: Optional[str] = None) -> Dict:
        if version:
            version = version.lstrip("^~>=<") # Simple version normalization

        url = f"{self.npm_registry_url}{package_name}"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if version and version in data.get('versio'):
                version_data = data['versions'][version]
            else:
                latest_version = data.get('dist-tags', {}).get('latest')
                version_data = data['versions'].get(latest_version, {}) 

            return {
                "name": package_name,
                "version": version,
                "description": version_data.get('description', ''),
                "homepage": version_data.get('homepage', ''),
                "repository": version_data.get('repository', {}),
                "keywords": version_data.get('keywords', []),
                "readme": version_data.get('readme', 'No README available.'),
                "dependencies": version_data.get('dependencies', {}),
                "dist_tags": data.get('dist-tags', {}),
                "license": version_data.get('license', 'Unknown'),
                "time_updated": data.get('time', {}).get(version, '')
            }
        except requests.RequestException as e:
            return {
                "name": package_name,
                "error": f"Failed to fetch package info: {str(e)}"
            }
        

    def fetch_multiple_packages(self, packages: Dict[str, Optional[str]], delay: float=0.1) -> Dict[str, Dict]:
        results = {}
        for pkg, ver in packages.items():
            results[pkg] = self.fetch_package_info(pkg, ver)
            if delay > 0:
              time.sleep(delay)  # To avoid hitting rate limits
        return results
    
    def get_package_types(self, package_name: str) -> Optional[str]:
        """
        Get the TypeScript packge for a given package, if it exists.
        """
        types_package = f"@types/{package_name.replace('@', '').replace('/', '__')}"
        url = f"{self.npm_registry_url}/{types_package}"
        try:
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                return types_package
        except requests.RequestException:
            pass
        return None
    
    