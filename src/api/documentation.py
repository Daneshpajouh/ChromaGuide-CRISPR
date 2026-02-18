"""
API documentation and OpenAPI/Swagger integration.

Features:
- FastAPI integration
- Swagger/OpenAPI schema
- Endpoint documentation
- Request/response validation
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json


@dataclass
class APIEndpoint:
    """API endpoint specification."""
    path: str
    method: str
    summary: str
    description: str
    parameters: Dict[str, Any] = None
    request_body: Dict[str, Any] = None
    response_schema: Dict[str, Any] = None
    tags: List[str] = None


class OpenAPIGenerator:
    """Generate OpenAPI/Swagger documentation."""
    
    def __init__(self, title: str = "CRISPR API", version: str = "1.0.0"):
        self.title = title
        self.version = version
        self.endpoints = []
        self.schemas = {}
    
    def add_endpoint(self, endpoint: APIEndpoint):
        """Add endpoint documentation."""
        self.endpoints.append(endpoint)
    
    def add_schema(self, name: str, schema: Dict):
        """Add reusable schema."""
        self.schemas[name] = schema
    
    def generate_spec(self) -> Dict:
        """Generate OpenAPI specification."""
        paths = {}
        
        for endpoint in self.endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}
            
            method = endpoint.method.lower()
            
            paths[endpoint.path][method] = {
                'summary': endpoint.summary,
                'description': endpoint.description,
                'tags': endpoint.tags or [],
                'responses': {
                    '200': {
                        'description': 'Success',
                        'content': {
                            'application/json': {
                                'schema': endpoint.response_schema or {}
                            }
                        }
                    },
                    '400': {'description': 'Bad Request'},
                    '500': {'description': 'Server Error'}
                }
            }
            
            if endpoint.request_body:
                paths[endpoint.path][method]['requestBody'] = {
                    'content': {
                        'application/json': {
                            'schema': endpoint.request_body
                        }
                    }
                }
            
            if endpoint.parameters:
                paths[endpoint.path][method]['parameters'] = endpoint.parameters
        
        spec = {
            'openapi': '3.0.0',
            'info': {
                'title': self.title,
                'version': self.version
            },
            'paths': paths,
            'components': {
                'schemas': self.schemas
            }
        }
        
        return spec
    
    def save_spec(self, filepath: Path):
        """Save OpenAPI spec to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        spec = self.generate_spec()
        with open(filepath, 'w') as f:
            json.dump(spec, f, indent=2)


class FastAPIDocBuilder:
    """Build FastAPI documentation."""
    
    @staticmethod
    def create_prediction_endpoint():
        """Example prediction endpoint."""
        example_code = """
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(
    title="CRISPR Prediction API",
    description="API for predicting CRISPR on-target efficiency",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    sequence: str
    guide_rna: str
    
class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    '''Predict CRISPR efficiency for given sequence.'''
    # Model inference
    prediction = 0.85  # Placeholder
    confidence = 0.92
    
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        model_version="v3.0"
    )

@app.get("/health")
def health_check():
    '''Health check endpoint.'''
    return {"status": "healthy", "version": "v3.0"}
"""
        return example_code
    
    @staticmethod
    def create_batch_endpoint():
        """Example batch prediction endpoint."""
        example_code = """
@app.post("/predict-batch")
def predict_batch(sequences: List[str]):
    '''Predict CRISPR efficiency for multiple sequences.'''
    predictions = []
    for seq in sequences:
        pred = model.predict(seq)  # Placeholder
        predictions.append({
            "sequence": seq,
            "prediction": float(pred),
            "confidence": 0.9
        })
    return {"predictions": predictions}
"""
        return example_code


class DocumentationGenerator:
    """Generate API documentation."""
    
    @staticmethod
    def generate_markdown_docs(endpoints: List[APIEndpoint]) -> str:
        """Generate Markdown documentation."""
        doc = "# CRISPR Prediction API Documentation\n\n"
        
        for endpoint in endpoints:
            doc += f"## {endpoint.summary}\n\n"
            doc += f"**Endpoint:** `{endpoint.method.upper()} {endpoint.path}`\n\n"
            doc += f"**Description:** {endpoint.description}\n\n"
            
            if endpoint.parameters:
                doc += "### Parameters\n\n"
                for param_name, param_spec in endpoint.parameters.items():
                    doc += f"- `{param_name}`: {param_spec.get('description', 'N/A')}\n"
                doc += "\n"
            
            if endpoint.request_body:
                doc += "### Request Body\n\n"
                doc += "```json\n"
                doc += json.dumps(endpoint.request_body, indent=2)
                doc += "\n```\n\n"
            
            if endpoint.response_schema:
                doc += "### Response\n\n"
                doc += "```json\n"
                doc += json.dumps(endpoint.response_schema, indent=2)
                doc += "\n```\n\n"
        
        return doc
    
    @staticmethod
    def save_documentation(docs: str, filepath: Path):
        """Save documentation to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(docs)


class ClientCodeGenerator:
    """Generate client code for API."""
    
    @staticmethod
    def generate_python_client(spec: Dict) -> str:
        """Generate Python client code."""
        code = '''
import requests
from typing import List, Dict

class CRISPRPredictionClient:
    """Python client for CRISPR Prediction API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def predict(self, sequence: str, guide_rna: str) -> Dict:
        """Make single prediction."""
        response = requests.post(
            f"{self.base_url}/predict",
            json={"sequence": sequence, "guide_rna": guide_rna}
        )
        return response.json()
    
    def predict_batch(self, sequences: List[str]) -> Dict:
        """Make batch predictions."""
        response = requests.post(
            f"{self.base_url}/predict-batch",
            json={"sequences": sequences}
        )
        return response.json()
    
    def health_check(self) -> Dict:
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Usage example
if __name__ == "__main__":
    client = CRISPRPredictionClient()
    
    result = client.predict(
        sequence="ATGCGATCGATCGATCG",
        guide_rna="TCGATCGATC"
    )
    print(result)
'''
        return code
    
    @staticmethod
    def generate_javascript_client(spec: Dict) -> str:
        """Generate JavaScript client code."""
        code = '''
class CRISPRPredictionClient {
    constructor(baseUrl = "http://localhost:8000") {
        this.baseUrl = baseUrl;
    }
    
    async predict(sequence, guideRna) {
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                sequence: sequence,
                guide_rna: guideRna
            })
        });
        return await response.json();
    }
    
    async predictBatch(sequences) {
        const response = await fetch(`${this.baseUrl}/predict-batch`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({sequences: sequences})
        });
        return await response.json();
    }
    
    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`);
        return await response.json();
    }
}

// Usage example
const client = new CRISPRPredictionClient();
client.predict("ATGCGATCGATCGATCG", "TCGATCGATC").then(console.log);
'''
        return code
