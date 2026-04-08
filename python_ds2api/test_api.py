#!/usr/bin/env python3
"""
Test script for DS2API Python version.
Run this after starting the server to verify the API endpoints.
"""
import httpx
import json
import sys


BASE_URL = "http://localhost:5001"


def test_health():
    """Test health check endpoint."""
    print("Testing /healthz...")
    resp = httpx.get(f"{BASE_URL}/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    print("  ✓ Health check passed")


def test_root():
    """Test root endpoint."""
    print("Testing /...")
    resp = httpx.get(f"{BASE_URL}/")
    assert resp.status_code == 200
    data = resp.json()
    assert "name" in data
    assert "endpoints" in data
    print(f"  ✓ Root endpoint: {data['name']} v{data['version']}")


def test_models():
    """Test models endpoint."""
    print("Testing /v1/models...")
    resp = httpx.get(f"{BASE_URL}/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    print(f"  ✓ Found {len(data['data'])} models")
    for model in data["data"][:3]:
        print(f"    - {model['id']}")


def test_chat_completions_no_key():
    """Test chat completions without API key (should fail)."""
    print("Testing /v1/chat/completions without API key...")
    resp = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hello"}]
        }
    )
    assert resp.status_code == 401
    print("  ✓ Correctly returned 401 Unauthorized")


def test_chat_completions_with_invalid_key():
    """Test chat completions with invalid API key."""
    print("Testing /v1/chat/completions with invalid API key...")
    resp = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Authorization": "Bearer invalid-key"},
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hello"}]
        },
        timeout=30.0
    )
    # Should work with direct token mode (even if token is invalid)
    # The actual error will come from DeepSeek API
    print(f"  Response status: {resp.status_code}")


def test_embeddings():
    """Test embeddings endpoint."""
    print("Testing /v1/embeddings...")
    resp = httpx.post(
        f"{BASE_URL}/v1/embeddings",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "text-embedding-ada-002",
            "input": "Hello world"
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert len(data["data"][0]["embedding"]) == 1536
    print("  ✓ Embeddings returned 1536-dim vector")


def test_model_alias():
    """Test model alias resolution."""
    print("Testing model alias resolution...")
    resp = httpx.get(f"{BASE_URL}/v1/models/gpt-4o")
    # gpt-4o should be aliased to deepseek-chat
    if resp.status_code == 200:
        data = resp.json()
        print(f"  ✓ Model alias gpt-4o -> {data['id']}")
    else:
        print("  Note: gpt-4o alias not configured")


def main():
    """Run all tests."""
    print("=" * 50)
    print("DS2API Python - API Tests")
    print("=" * 50)
    print()

    tests = [
        test_health,
        test_root,
        test_models,
        test_chat_completions_no_key,
        test_embeddings,
        test_model_alias,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ Assertion failed: {e}")
            failed += 1
        except httpx.ConnectError:
            print(f"  ✗ Cannot connect to server at {BASE_URL}")
            print("  Please start the server first: python main.py")
            sys.exit(1)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
        print()

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
