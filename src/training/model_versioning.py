"""
Advanced model versioning and experiment management.

Features:
- Model versioning with Git
- Experiment metadata tracking
- Artifact management
- Model lineage tracking
"""

import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class ModelVersion:
    """Model version metadata."""
    name: str
    version: str
    timestamp: str
    git_commit: str
    parent_version: Optional[str] = None
    metrics: Dict[str, float] = None
    hyperparameters: Dict[str, Any] = None
    description: str = ""
    tags: List[str] = None


class ModelRegistry:
    """Manage model versions."""
    
    def __init__(self, registry_dir: Path = Path("models/registry")):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.versions = []
    
    def register_model(
        self,
        model_path: Path,
        version_info: ModelVersion
    ) -> str:
        """Register new model version."""
        version_dir = self.registry_dir / version_info.name / version_info.version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model artifact
        import shutil
        artifact_path = version_dir / "model"
        if Path(model_path).is_file():
            shutil.copy(model_path, artifact_path)
        else:
            shutil.copytree(model_path, artifact_path)
        
        # Save metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(version_info), f, indent=2, default=str)
        
        self.versions.append(version_info)
        return str(version_dir)
    
    def get_model_version(self, name: str, version: str) -> Optional[Path]:
        """Get model by version."""
        model_path = self.registry_dir / name / version / "model"
        if model_path.exists():
            return model_path
        return None
    
    def list_model_versions(self, name: str) -> List[ModelVersion]:
        """List all versions of a model."""
        model_dir = self.registry_dir / name
        versions = []
        
        if model_dir.exists():
            for version_dir in model_dir.iterdir():
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        versions.append(ModelVersion(**metadata))
        
        return sorted(versions, key=lambda v: v.timestamp, reverse=True)
    
    def get_latest_version(self, name: str) -> Optional[ModelVersion]:
        """Get latest version of model."""
        versions = self.list_model_versions(name)
        return versions[0] if versions else None
    
    def tag_version(self, name: str, version: str, tag: str):
        """Add tag to model version."""
        metadata_path = self.registry_dir / name / version / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if 'tags' not in metadata:
                metadata['tags'] = []
            if tag not in metadata['tags']:
                metadata['tags'].append(tag)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)


class ModelLineage:
    """Track model training lineage."""
    
    def __init__(self):
        self.lineage = {}
    
    def add_relationship(
        self,
        parent_model: str,
        child_model: str,
        transformation: str
    ):
        """Record parent-child relationship."""
        if parent_model not in self.lineage:
            self.lineage[parent_model] = []
        
        self.lineage[parent_model].append({
            'child': child_model,
            'transformation': transformation,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_ancestry(self, model_name: str) -> List[str]:
        """Get model ancestry."""
        ancestry = [model_name]
        
        # Reverse lookup
        for parent, children in self.lineage.items():
            for child in children:
                if child['child'] == ancestry[-1]:
                    ancestry.append(parent)
                    break
        
        return ancestry
    
    def export_lineage(self, filepath: Path):
        """Export lineage to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.lineage, f, indent=2)


class ArtifactManager:
    """Manage training artifacts."""
    
    def __init__(self, artifacts_dir: Path = Path("artifacts")):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def save_artifact(
        self,
        artifact_path: Path,
        artifact_type: str,
        experiment_id: str
    ) -> Path:
        """Save artifact with metadata."""
        exp_dir = self.artifacts_dir / experiment_id / artifact_type
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate checksum
        with open(artifact_path, 'rb') as f:
            checksum = hashlib.md5(f.read()).hexdigest()
        
        # Save artifact and metadata
        import shutil
        dest_path = exp_dir / artifact_path.name
        shutil.copy(artifact_path, dest_path)
        
        metadata = {
            'original_path': str(artifact_path),
            'saved_path': str(dest_path),
            'checksum': checksum,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = exp_dir / f"{artifact_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return dest_path
    
    def retrieve_artifact(
        self,
        experiment_id: str,
        artifact_type: str,
        artifact_name: str
    ) -> Optional[Path]:
        """Retrieve artifact."""
        artifact_path = self.artifacts_dir / experiment_id / artifact_type / artifact_name
        if artifact_path.exists():
            return artifact_path
        return None
    
    def list_artifacts(self, experiment_id: str) -> Dict[str, List[str]]:
        """List all artifacts for experiment."""
        exp_dir = self.artifacts_dir / experiment_id
        artifacts = {}
        
        if exp_dir.exists():
            for type_dir in exp_dir.iterdir():
                if type_dir.is_dir():
                    artifacts[type_dir.name] = [
                        f.name for f in type_dir.iterdir()
                        if f.suffix != '.json'
                    ]
        
        return artifacts


class ExperimentTrackingRegistry:
    """Central registry for all experiments."""
    
    def __init__(self, registry_file: Path = Path("experiments_registry.json")):
        self.registry_file = Path(registry_file)
        self.experiments = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def register_experiment(
        self,
        experiment_id: str,
        metadata: Dict[str, Any]
    ):
        """Register new experiment."""
        self.experiments[experiment_id] = {
            'metadata': metadata,
            'timestamp': datetime.now().isoformat(),
            'status': 'running'
        }
        self._save_registry()
    
    def update_experiment(self, experiment_id: str, updates: Dict):
        """Update experiment metadata."""
        if experiment_id in self.experiments:
            self.experiments[experiment_id].update(updates)
            self._save_registry()
    
    def _save_registry(self):
        """Save registry to file."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment metadata."""
        return self.experiments.get(experiment_id)
    
    def list_experiments(self, status: Optional[str] = None) -> List[str]:
        """List experiments with optional status filter."""
        if status:
            return [
                exp_id for exp_id, exp in self.experiments.items()
                if exp.get('status') == status
            ]
        return list(self.experiments.keys())
