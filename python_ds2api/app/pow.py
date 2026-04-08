"""
DeepSeek PoW (Proof of Work) implementation in Python.
DeepSeekHashV1 = SHA3-256 but skips round 0 (only does rounds 1..23).
Optimized with numba JIT compilation for ~100x speedup.
"""
import base64
import json
import struct
from typing import Optional, Dict, Any

# Try to import numba for JIT compilation
try:
    from numba import njit, prange
    import numpy as np
    HAS_NUMBA = True
    print("[pow] numba JIT available - using optimized implementation")
except ImportError:
    HAS_NUMBA = False
    print("[pow] numba not available - using pure Python (slower)")

# Round constants for Keccak-f[1600]
RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
]


def rotl64(v: int, k: int) -> int:
    """Rotate left a 64-bit integer."""
    return ((v << k) | (v >> (64 - k))) & 0xFFFFFFFFFFFFFFFF


def keccak_f23(s: list) -> None:
    """
    Keccak-f[1600] permutation but skipping round 0.
    DeepSeekHashV1 uses rounds 1..23 only.
    Directly ported from Go implementation.
    """
    a0, a1, a2, a3, a4 = s[0], s[1], s[2], s[3], s[4]
    a5, a6, a7, a8, a9 = s[5], s[6], s[7], s[8], s[9]
    a10, a11, a12, a13, a14 = s[10], s[11], s[12], s[13], s[14]
    a15, a16, a17, a18, a19 = s[15], s[16], s[17], s[18], s[19]
    a20, a21, a22, a23, a24 = s[20], s[21], s[22], s[23], s[24]

    for r in range(1, 24):
        # Theta step
        c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20
        c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21
        c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22
        c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23
        c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24

        d0 = c4 ^ rotl64(c1, 1)
        d1 = c0 ^ rotl64(c2, 1)
        d2 = c1 ^ rotl64(c3, 1)
        d3 = c2 ^ rotl64(c4, 1)
        d4 = c3 ^ rotl64(c0, 1)

        a0 ^= d0
        a5 ^= d0
        a10 ^= d0
        a15 ^= d0
        a20 ^= d0
        a1 ^= d1
        a6 ^= d1
        a11 ^= d1
        a16 ^= d1
        a21 ^= d1
        a2 ^= d2
        a7 ^= d2
        a12 ^= d2
        a17 ^= d2
        a22 ^= d2
        a3 ^= d3
        a8 ^= d3
        a13 ^= d3
        a18 ^= d3
        a23 ^= d3
        a4 ^= d4
        a9 ^= d4
        a14 ^= d4
        a19 ^= d4
        a24 ^= d4

        # Rho and Pi steps
        b0 = a0
        b10 = rotl64(a1, 1)
        b20 = rotl64(a2, 62)
        b5 = rotl64(a3, 28)
        b15 = rotl64(a4, 27)
        b16 = rotl64(a5, 36)
        b1 = rotl64(a6, 44)
        b11 = rotl64(a7, 6)
        b21 = rotl64(a8, 55)
        b6 = rotl64(a9, 20)
        b7 = rotl64(a10, 3)
        b17 = rotl64(a11, 10)
        b2 = rotl64(a12, 43)
        b12 = rotl64(a13, 25)
        b22 = rotl64(a14, 39)
        b23 = rotl64(a15, 41)
        b8 = rotl64(a16, 45)
        b18 = rotl64(a17, 15)
        b3 = rotl64(a18, 21)
        b13 = rotl64(a19, 8)
        b14 = rotl64(a20, 18)
        b24 = rotl64(a21, 2)
        b9 = rotl64(a22, 61)
        b19 = rotl64(a23, 56)
        b4 = rotl64(a24, 14)

        # Chi step
        a0 = b0 ^ ((~b1) & b2)
        a1 = b1 ^ ((~b2) & b3)
        a2 = b2 ^ ((~b3) & b4)
        a3 = b3 ^ ((~b4) & b0)
        a4 = b4 ^ ((~b0) & b1)
        a5 = b5 ^ ((~b6) & b7)
        a6 = b6 ^ ((~b7) & b8)
        a7 = b7 ^ ((~b8) & b9)
        a8 = b8 ^ ((~b9) & b5)
        a9 = b9 ^ ((~b5) & b6)
        a10 = b10 ^ ((~b11) & b12)
        a11 = b11 ^ ((~b12) & b13)
        a12 = b12 ^ ((~b13) & b14)
        a13 = b13 ^ ((~b14) & b10)
        a14 = b14 ^ ((~b10) & b11)
        a15 = b15 ^ ((~b16) & b17)
        a16 = b16 ^ ((~b17) & b18)
        a17 = b17 ^ ((~b18) & b19)
        a18 = b18 ^ ((~b19) & b15)
        a19 = b19 ^ ((~b15) & b16)
        a20 = b20 ^ ((~b21) & b22)
        a21 = b21 ^ ((~b22) & b23)
        a22 = b22 ^ ((~b23) & b24)
        a23 = b23 ^ ((~b24) & b20)
        a24 = b24 ^ ((~b20) & b21)

        # Iota step
        a0 ^= RC[r]

    s[0], s[1], s[2], s[3], s[4] = a0, a1, a2, a3, a4
    s[5], s[6], s[7], s[8], s[9] = a5, a6, a7, a8, a9
    s[10], s[11], s[12], s[13], s[14] = a10, a11, a12, a13, a14
    s[15], s[16], s[17], s[18], s[19] = a15, a16, a17, a18, a19
    s[20], s[21], s[22], s[23], s[24] = a20, a21, a22, a23, a24


def deepseek_hash_v1(data: bytes) -> bytes:
    """
    Compute DeepSeekHashV1 hash.
    This is SHA3-256 but skipping round 0 of Keccak-f[1600].
    """
    rate = 136  # 1088 bits / 8

    # Initialize state (25 x 64-bit words, all zero)
    s = [0] * 25

    # Process full blocks
    off = 0
    while off + rate <= len(data):
        for i in range(rate // 8):
            s[i] ^= struct.unpack('<Q', data[off + i * 8:off + (i + 1) * 8])[0]
        keccak_f23(s)
        off += rate

    # Final block with padding
    final = bytearray(rate)
    remaining = len(data) - off
    final[:remaining] = data[off:]
    # SHA3 padding: append 0x06 at the end of the data, then pad with zeros,
    # then set the last byte to 0x80
    final[remaining] = 0x06  # SHA3 domain separator - AFTER the data
    final[rate - 1] |= 0x80  # Set high bit of last byte

    for i in range(rate // 8):
        s[i] ^= struct.unpack('<Q', final[i * 8:(i + 1) * 8])[0]
    keccak_f23(s)

    # Output first 32 bytes (256 bits)
    out = bytearray(32)
    for i in range(4):
        struct.pack_into('<Q', out, i * 8, s[i])

    return bytes(out)


def build_prefix(salt: str, expire_at: int) -> str:
    """Build the prefix string for PoW computation."""
    return f"{salt}_{expire_at}_"


# Numba-optimized functions (compiled on first use)
if HAS_NUMBA:
    RC_ARRAY = np.array([
        0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
        0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
        0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
        0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
        0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
        0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
        0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
        0x8000000000008080, 0x0000000080000001, 0x8000000080008008
    ], dtype=np.uint64)

    @njit
    def rotl64_numba(v, k):
        return ((v << k) | (v >> (64 - k))) & 0xFFFFFFFFFFFFFFFF

    @njit
    def keccak_f23_numba(s):
        a0, a1, a2, a3, a4 = s[0], s[1], s[2], s[3], s[4]
        a5, a6, a7, a8, a9 = s[5], s[6], s[7], s[8], s[9]
        a10, a11, a12, a13, a14 = s[10], s[11], s[12], s[13], s[14]
        a15, a16, a17, a18, a19 = s[15], s[16], s[17], s[18], s[19]
        a20, a21, a22, a23, a24 = s[20], s[21], s[22], s[23], s[24]

        for r in range(1, 24):
            c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20
            c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21
            c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22
            c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23
            c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24

            d0 = c4 ^ rotl64_numba(c1, 1)
            d1 = c0 ^ rotl64_numba(c2, 1)
            d2 = c1 ^ rotl64_numba(c3, 1)
            d3 = c2 ^ rotl64_numba(c4, 1)
            d4 = c3 ^ rotl64_numba(c0, 1)

            a0 ^= d0; a5 ^= d0; a10 ^= d0; a15 ^= d0; a20 ^= d0
            a1 ^= d1; a6 ^= d1; a11 ^= d1; a16 ^= d1; a21 ^= d1
            a2 ^= d2; a7 ^= d2; a12 ^= d2; a17 ^= d2; a22 ^= d2
            a3 ^= d3; a8 ^= d3; a13 ^= d3; a18 ^= d3; a23 ^= d3
            a4 ^= d4; a9 ^= d4; a14 ^= d4; a19 ^= d4; a24 ^= d4

            b0 = a0
            b10 = rotl64_numba(a1, 1)
            b20 = rotl64_numba(a2, 62)
            b5 = rotl64_numba(a3, 28)
            b15 = rotl64_numba(a4, 27)
            b16 = rotl64_numba(a5, 36)
            b1 = rotl64_numba(a6, 44)
            b11 = rotl64_numba(a7, 6)
            b21 = rotl64_numba(a8, 55)
            b6 = rotl64_numba(a9, 20)
            b7 = rotl64_numba(a10, 3)
            b17 = rotl64_numba(a11, 10)
            b2 = rotl64_numba(a12, 43)
            b12 = rotl64_numba(a13, 25)
            b22 = rotl64_numba(a14, 39)
            b23 = rotl64_numba(a15, 41)
            b8 = rotl64_numba(a16, 45)
            b18 = rotl64_numba(a17, 15)
            b3 = rotl64_numba(a18, 21)
            b13 = rotl64_numba(a19, 8)
            b14 = rotl64_numba(a20, 18)
            b24 = rotl64_numba(a21, 2)
            b9 = rotl64_numba(a22, 61)
            b19 = rotl64_numba(a23, 56)
            b4 = rotl64_numba(a24, 14)

            a0 = b0 ^ ((~b1) & b2)
            a1 = b1 ^ ((~b2) & b3)
            a2 = b2 ^ ((~b3) & b4)
            a3 = b3 ^ ((~b4) & b0)
            a4 = b4 ^ ((~b0) & b1)
            a5 = b5 ^ ((~b6) & b7)
            a6 = b6 ^ ((~b7) & b8)
            a7 = b7 ^ ((~b8) & b9)
            a8 = b8 ^ ((~b9) & b5)
            a9 = b9 ^ ((~b5) & b6)
            a10 = b10 ^ ((~b11) & b12)
            a11 = b11 ^ ((~b12) & b13)
            a12 = b12 ^ ((~b13) & b14)
            a13 = b13 ^ ((~b14) & b10)
            a14 = b14 ^ ((~b10) & b11)
            a15 = b15 ^ ((~b16) & b17)
            a16 = b16 ^ ((~b17) & b18)
            a17 = b17 ^ ((~b18) & b19)
            a18 = b18 ^ ((~b19) & b15)
            a19 = b19 ^ ((~b15) & b16)
            a20 = b20 ^ ((~b21) & b22)
            a21 = b21 ^ ((~b22) & b23)
            a22 = b22 ^ ((~b23) & b24)
            a23 = b23 ^ ((~b24) & b20)
            a24 = b24 ^ ((~b20) & b21)

            a0 ^= RC_ARRAY[r]

        s[0], s[1], s[2], s[3], s[4] = a0, a1, a2, a3, a4
        s[5], s[6], s[7], s[8], s[9] = a5, a6, a7, a8, a9
        s[10], s[11], s[12], s[13], s[14] = a10, a11, a12, a13, a14
        s[15], s[16], s[17], s[18], s[19] = a15, a16, a17, a18, a19
        s[20], s[21], s[22], s[23], s[24] = a20, a21, a22, a23, a24

    @njit
    def deepseek_hash_v1_numba(data):
        rate = 136
        s = np.zeros(25, dtype=np.uint64)

        off = 0
        while off + rate <= len(data):
            for i in range(rate // 8):
                val = 0
                for j in range(8):
                    val |= data[off + i * 8 + j] << (j * 8)
                s[i] ^= val
            keccak_f23_numba(s)
            off += rate

        final = np.zeros(rate, dtype=np.uint8)
        remaining = len(data) - off
        for i in range(remaining):
            final[i] = data[off + i]
        final[remaining] = 0x06
        final[rate - 1] |= 0x80

        for i in range(rate // 8):
            val = 0
            for j in range(8):
                val |= final[i * 8 + j] << (j * 8)
            s[i] ^= val
        keccak_f23_numba(s)

        out = np.zeros(32, dtype=np.uint8)
        for i in range(4):
            val = s[i]
            for j in range(8):
                out[i * 8 + j] = (val >> (j * 8)) & 0xFF
        return out

    @njit
    def solve_pow_numba(prefix_bytes, target_bytes, difficulty):
        prefix_len = len(prefix_bytes)
        for nonce in range(difficulty):
            # Convert nonce to string bytes manually
            if nonce == 0:
                nonce_len = 1
                # Build data = prefix + '0'
                data = np.zeros(prefix_len + 1, dtype=np.uint8)
                for i in range(prefix_len):
                    data[i] = prefix_bytes[i]
                data[prefix_len] = 48  # ord('0')
            else:
                # Count digits first
                n = nonce
                nonce_len = 0
                temp = n
                while temp > 0:
                    nonce_len += 1
                    temp //= 10

                # Build data = prefix + nonce_str
                data = np.zeros(prefix_len + nonce_len, dtype=np.uint8)
                for i in range(prefix_len):
                    data[i] = prefix_bytes[i]

                # Fill digits from right to left
                n = nonce
                for i in range(nonce_len - 1, -1, -1):
                    data[prefix_len + i] = 48 + (n % 10)  # ord('0') + digit
                    n //= 10

            result = deepseek_hash_v1_numba(data)

            # Compare with target
            match = True
            for i in range(32):
                if result[i] != target_bytes[i]:
                    match = False
                    break

            if match:
                return nonce
        return -1


def solve_pow(challenge_hex: str, salt: str, expire_at: int, difficulty: int = 144000) -> Optional[int]:
    """
    Solve the PoW challenge.
    Find nonce in [0, difficulty) such that DeepSeekHashV1(prefix + str(nonce)) == challenge.
    """
    if len(challenge_hex) != 64:
        raise ValueError("Challenge must be 64 hex characters")

    target = bytes.fromhex(challenge_hex)
    prefix = build_prefix(salt, expire_at).encode('utf-8')

    print(f"[solve_pow] Searching for nonce, prefix: {prefix.decode()}, difficulty: {difficulty}")

    if HAS_NUMBA:
        # Use numba-optimized version
        prefix_arr = np.frombuffer(prefix, dtype=np.uint8)
        target_arr = np.frombuffer(target, dtype=np.uint8)

        result = solve_pow_numba(prefix_arr, target_arr, difficulty)
        if result >= 0:
            print(f"[solve_pow] Found nonce: {result}")
            return result
        print(f"[solve_pow] No solution found within difficulty {difficulty}")
        return None
    else:
        # Pure Python fallback
        for nonce in range(difficulty):
            data = prefix + str(nonce).encode('utf-8')
            result = deepseek_hash_v1(data)

            if result == target:
                print(f"[solve_pow] Found nonce: {nonce}")
                return nonce

            if nonce > 0 and nonce % 10000 == 0:
                print(f"[solve_pow] Progress: {nonce}/{difficulty}")

        print(f"[solve_pow] No solution found within difficulty {difficulty}")
        return None


class Challenge:
    """PoW challenge from DeepSeek API."""

    def __init__(self, data: Dict[str, Any]):
        self.algorithm: str = data.get("algorithm", "")
        self.challenge: str = data.get("challenge", "")
        self.salt: str = data.get("salt", "")
        self.expire_at: int = int(data.get("expire_at", 1680000000) or 1680000000)
        self.difficulty: int = int(data.get("difficulty", 144000) or 144000)
        self.signature: str = data.get("signature", "")
        self.target_path: str = data.get("target_path", "")

        print(f"[Challenge] algorithm={self.algorithm}, salt={self.salt}, "
              f"expire_at={self.expire_at}, difficulty={self.difficulty}")

    def solve(self) -> Optional[int]:
        """Solve this challenge and return the answer."""
        if self.algorithm != "DeepSeekHashV1":
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        difficulty = self.difficulty if self.difficulty > 0 else 144000
        return solve_pow(self.challenge, self.salt, self.expire_at, difficulty)

    def build_pow_header(self, answer: int) -> str:
        """Build the x-ds-pow-response header value."""
        header_data = {
            "algorithm": self.algorithm,
            "challenge": self.challenge,
            "salt": self.salt,
            "answer": answer,
            "signature": self.signature,
            "target_path": self.target_path
        }
        json_str = json.dumps(header_data, separators=(',', ':'))
        return base64.b64encode(json_str.encode('utf-8')).decode('utf-8')

    def solve_and_build_header(self) -> Optional[str]:
        """Solve the challenge and build the header in one step."""
        answer = self.solve()
        if answer is None:
            return None
        return self.build_pow_header(answer)


if __name__ == "__main__":
    # Test
    test_data = b"test"
    result = deepseek_hash_v1(test_data)
    print(f"DeepSeekHashV1('test') = {result.hex()}")
