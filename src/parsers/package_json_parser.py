import json
from pathlib import Path
from typing import Dict, List, Optional

class PackageJsonParser:
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)

    def parse_package_json(self, file_path: Optional[str] = None) -> Dict[str, str]:
        # If no path is provided, use project_root/package.json
        # Otherwise, use provided path
        if file_path is None:
            file_path = self.project_root / "package.json"
        else:
            file_path = Path(file_path)
        
        # if filepath does not exist, raise error
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist.")
        
        # Parse package.json for dependencies
        with open(file_path, 'r') as f:
            data = json.load(f)

        deps = {}

        for dep_type in ['dependencies', 'devDependencies', 'peerDependencies']:
            if dep_type in data:
                deps.update(data[dep_type])

        return deps
    
    def parse_package_lock(self, file_path: Optional[str] = None) -> Dict[str, str]:
        # If no path is provided, use project_root/package-lock.json
        # Otherwise, use provided path
        if file_path is None:
            file_path = self.project_root / "package-lock.json"
        else:
            file_path = Path(file_path)
        
        # if filepath does not exist, raise error
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist.")
        
        # Parse package-lock.json for dependencies
        with open(file_path, 'r') as f:
            data = json.load(f)

        packages = {}

        # check if "packages" key exists in package-lock.json (npm 7+)
        # if it does, then get the path and info from the item and see if it is a node module. If it is, add to dict with version, resolved, and integrity information.
        if 'packages' in data:
            for pkg_path, pkg_info in data['packages'].items():
                if pkg_path and pkg_path.startswith("node_modules/"):
                    pkg_name = pkg_path.split("node_modules/")[1]
                    packages[pkg_name] = {
                        "version": pkg_info.get("version"),
                        "resolved": pkg_info.get("resolved"),
                        "integrity": pkg_info.get("integrity")
                    }
        elif 'dependencies' in data:
            # Fallback for older npm versions
            for pkg_name, pkg_info in data['dependencies'].items():
                packages[pkg_name] = {
                    "version": pkg_info.get("version"),
                    "resolved": pkg_info.get("resolved"),
                    "integrity": pkg_info.get("integrity")
                }

        return packages
    
    def get_all_dependencies(self) -> Dict[str, Dict[str, Optional[str]]]:
        pkg_deps = self.parse_package_json()
        lock_deps = self.parse_package_lock()

        all_deps = {}
        for dep, version_spec in pkg_deps.items():
            # print(dep)
            all_deps[dep] = {
                "version_spec": version_spec,
                "resolved_version": lock_deps.get(dep, {}).get("version", version_spec),
            }

        for dep, lock_info in lock_deps.items():
            all_deps[dep] = {
                "version_spec": None,
                "resolved_version": lock_info.get("version"),
            }


        return all_deps