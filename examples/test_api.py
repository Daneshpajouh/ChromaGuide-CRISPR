#!/usr/bin/env python3
"""
Example: Running ChromaGuide API and testing endpoints

This script demonstrates:
1. Starting the API server
2. Testing all endpoints
3. Performance benchmarking
4. Error handling
"""

import subprocess
import time
import requests
import json
import sys
from pathlib import Path

API_URL = "http://localhost:8000"
SERVER_PROCESS = None


def start_server():
    """Start the FastAPI server."""
    global SERVER_PROCESS
    
    print("Starting ChromaGuide API server...")
    SERVER_PROCESS = subprocess.Popen(
        ["uvicorn", "src.api.main:app", "--reload", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for server to start
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                print(f"✓ Server started successfully")
                return True
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                time.sleep(1)
                continue
    
    print(f"✗ Failed to start server after {max_retries} seconds")
    return False


def stop_server():
    """Stop the FastAPI server."""
    global SERVER_PROCESS
    
    if SERVER_PROCESS:
        print("Stopping server...")
        SERVER_PROCESS.terminate()
        SERVER_PROCESS.wait()
        print("✓ Server stopped")


def test_health_check():
    """Test health check endpoint."""
    print("\n" + "="*70)
    print("Testing: GET /health")
    print("="*70)
    
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        
        data = response.json()
        print(json.dumps(data, indent=2))
        
        return response.status_code == 200
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_single_prediction():
    """Test single prediction endpoint."""
    print("\n" + "="*70)
    print("Testing: POST /predict (single guideRNA)")
    print("="*70)
    
    request_data = {
        "sequence": "ACGTACGTACGTACGTACGTACG",
        "cas_type": "cas9",
        "include_uncertainty": True,
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(f"{API_URL}/predict", json=request_data)
        print(f"\nStatus: {response.status_code}")
        
        data = response.json()
        
        # Print key results
        print(f"\nResults:")
        print(f"  Efficiency Score: {data['efficiency_score']:.3f}")
        if data.get('efficiency_lower'):
            print(f"  Efficiency CI:    [{data['efficiency_lower']:.3f}, {data['efficiency_upper']:.3f}]")
        print(f"  Off-target Risk:  {data['off_target_risk']:.3f}")
        print(f"  Design Score:     {data['design_score']:.3f}")
        print(f"  Safety Tier:      {data['safety_tier']}")
        
        return response.status_code == 200
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\n" + "="*70)
    print("Testing: POST /predict/batch (multiple guideRNAs)")
    print("="*70)
    
    guides = [
        "ACGTACGTACGTACGTACGTACG",
        "TGCATGCATGCATGCATGCATGCA",
        "AAAATTTTCCCCGGGGACGTACGT",
        "GGGGCCCCAAAAGGGGCCCCAAAA",
        "TTTTAAAATTTTCCCCGGGGAAAA",
    ]
    
    request_data = {
        "guides": [{"sequence": seq} for seq in guides],
        "return_all": False,
    }
    
    print(f"Predicting {len(guides)} guides...")
    
    try:
        start_time = time.time()
        response = requests.post(f"{API_URL}/predict/batch", json=request_data)
        elapsed = time.time() - start_time
        
        print(f"Status: {response.status_code}")
        
        data = response.json()
        
        print(f"\nResults:")
        print(f"  Total Guides:       {data['num_guides']}")
        print(f"  High Quality:       {data['num_high_quality']}")
        print(f"  Processing Time:    {data['processing_time_sec']:.3f}s")
        print(f"  Throughput:         {data['num_guides']/data['processing_time_sec']:.0f} guides/sec")
        
        print(f"\nTop Guides:")
        if data.get('top_guides'):
            for i, pred in enumerate(data['top_guides'][:3], 1):
                print(f"  {i}. {pred['guide_sequence']}")
                print(f"     Design Score: {pred['design_score']:.3f}, Safety: {pred['safety_tier']}")
        
        return response.status_code == 200
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_models_info():
    """Test models info endpoint."""
    print("\n" + "="*70)
    print("Testing: POST /models/info")
    print("="*70)
    
    try:
        response = requests.post(f"{API_URL}/models/info", json={})
        print(f"Status: {response.status_code}")
        
        data = response.json()
        
        print(f"\nAvailable Models:")
        for model_name in data['available_models'][:5]:
            print(f"  - {model_name}")
        
        if len(data['available_models']) > 5:
            print(f"  ... and {len(data['available_models']) - 5} more")
        
        return response.status_code == 200
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_design_score():
    """Test design score endpoint."""
    print("\n" + "="*70)
    print("Testing: POST /design-score")
    print("="*70)
    
    request_data = {
        "efficiency_score": 0.78,
        "off_target_risk": 0.12,
        "specificity_score": 0.65,
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(f"{API_URL}/design-score", json=request_data)
        print(f"\nStatus: {response.status_code}")
        
        data = response.json()
        
        print(f"\nResults:")
        print(f"  Design Score: {data['design_score']:.3f}")
        print(f"  Components:")
        for key, val in data['components'].items():
            print(f"    {key}: {val:.3f}")
        
        return response.status_code == 200
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_error_handling():
    """Test error handling."""
    print("\n" + "="*70)
    print("Testing: Error Handling")
    print("="*70)
    
    # Test invalid sequence
    print("\n1. Invalid sequence (non-ACGT characters):")
    try:
        response = requests.post(f"{API_URL}/predict", json={
            "sequence": "ACGTXYZABC"
        })
        print(f"   Status: {response.status_code}")
        if response.status_code != 200:
            error = response.json()
            print(f"   Error: {error['error']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test missing required field
    print("\n2. Missing required field:")
    try:
        response = requests.post(f"{API_URL}/predict", json={})
        print(f"   Status: {response.status_code}")
        if response.status_code != 200:
            error = response.json()
            print(f"   Error: {error['error']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    return True


def benchmark():
    """Benchmark performance."""
    print("\n" + "="*70)
    print("Performance Benchmark")
    print("="*70)
    
    # Single prediction latency
    print("\n1. Single Prediction Latency (100 requests):")
    times = []
    for _ in range(100):
        start = time.time()
        requests.post(f"{API_URL}/predict", json={
            "sequence": "ACGTACGTACGTACGTACGTACG"
        })
        times.append(time.time() - start)
    
    import statistics
    print(f"   Mean:   {statistics.mean(times)*1000:.2f}ms")
    print(f"   Median: {statistics.median(times)*1000:.2f}ms")
    print(f"   Stdev:  {statistics.stdev(times)*1000:.2f}ms")
    
    # Batch prediction throughput
    print("\n2. Batch Prediction Throughput:")
    batch_sizes = [10, 50, 100]
    for batch_size in batch_sizes:
        guides = [{"sequence": "ACGTACGTACGTACGTACGTACG"} for _ in range(batch_size)]
        start = time.time()
        response = requests.post(f"{API_URL}/predict/batch", json={"guides": guides})
        elapsed = time.time() - start
        throughput = batch_size / elapsed
        print(f"   Batch {batch_size:3d}: {throughput:6.0f} guides/sec ({elapsed*1000:.1f}ms)")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ChromaGuide API Testing Suite")
    print("="*70)
    
    # Start server
    if not start_server():
        print("Failed to start server")
        sys.exit(1)
    
    time.sleep(2)  # Give server time to fully initialize
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Models Info", test_models_info),
        ("Design Score", test_design_score),
        ("Error Handling", test_error_handling),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Benchmark
    try:
        benchmark()
    except Exception as e:
        print(f"Benchmark failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for r in results.values() if r)
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    # Cleanup
    stop_server()
    
    # Exit with appropriate code
    sys.exit(0 if total_passed == total_tests else 1)


if __name__ == '__main__':
    main()
