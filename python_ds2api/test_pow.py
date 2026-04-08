#!/usr/bin/env python3
"""
Test script to verify DeepSeekHashV1 implementation.
Compare Python result with Go result.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pow import deepseek_hash_v1, solve_pow

def test_basic_hash():
    """Test basic hash computation."""
    print("=" * 50)
    print("Testing DeepSeekHashV1")
    print("=" * 50)

    # Test cases
    test_cases = [
        b"test",
        b"hello",
        b"",
        b"a",
        b"salt_12345_0",
    ]

    for data in test_cases:
        result = deepseek_hash_v1(data)
        print(f"DeepSeekHashV1({data!r}) = {result.hex()}")


def test_solve_known():
    """Test solving with a known prefix."""
    print("\n" + "=" * 50)
    print("Testing PoW Solve")
    print("=" * 50)

    # Test with a small prefix to find a solution quickly
    # We'll compute what nonce produces a specific hash
    salt = "test_salt"
    expire_at = 1234567890
    prefix = f"{salt}_{expire_at}_".encode()

    print(f"Prefix: {prefix}")

    # Compute first few hashes
    print("\nFirst few hashes:")
    for nonce in range(5):
        data = prefix + str(nonce).encode()
        h = deepseek_hash_v1(data)
        print(f"  nonce={nonce}: {h.hex()[:32]}...")

    # Try to solve with a small difficulty (create our own challenge)
    print("\nTesting solve with self-generated challenge:")
    for nonce in range(1000):
        data = prefix + str(nonce).encode()
        h = deepseek_hash_v1(data)
        # Use this hash as the challenge
        challenge_hex = h.hex()
        print(f"Using nonce {nonce} hash as challenge: {challenge_hex[:32]}...")

        result = solve_pow(challenge_hex, salt, expire_at, 1000)
        if result is not None:
            print(f"  -> Found solution: {result} (expected {nonce})")
            if result == nonce:
                print("  -> SUCCESS! Hash function works correctly!")
            else:
                print("  -> FAIL! Got different nonce!")
            break
        break


def test_go_comparison():
    """Compare with Go implementation."""
    print("\n" + "=" * 50)
    print("Comparison with Go implementation")
    print("=" * 50)

    # These are expected to match the Go output
    # We'll compute them and check if they match expected patterns

    test_data = b"test"
    result = deepseek_hash_v1(test_data)

    print(f"\nPython DeepSeekHashV1('test') = {result.hex()}")
    print("\nTo verify, run in Go:")
    print("  go run -ldflags=\"-extldflags=-Wl,-stack_size,0x1000000\" -v ./pow")
    print("\nOr create a test file:")
    print("""
package main

import (
    "encoding/hex"
    "fmt"
    "ds2api/pow"
)

func main() {
    data := []byte("test")
    result := pow.DeepSeekHashV1(data)
    fmt.Println("DeepSeekHashV1('test') =", hex.EncodeToString(result[:]))
}
""")


if __name__ == "__main__":
    test_basic_hash()
    test_solve_known()
    test_go_comparison()
