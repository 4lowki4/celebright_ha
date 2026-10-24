"""AWS Cognito SRP (Secure Remote Password) helper — dependency-free.

Celebright's Cognito app client only has USER_SRP_AUTH enabled (the default
for Amplify apps), so USER_PASSWORD_AUTH is rejected with
`InvalidParameterException: USER_PASSWORD_AUTH flow not enabled`. This module
implements the SRP-6a client side of Cognito's PASSWORD_VERIFIER challenge so
auth.py can log in with just email + password, no boto3/pycognito dependency.

The flow (orchestrated in auth.py):
  1. InitiateAuth USER_SRP_AUTH  → send SRP_A, receive SALT, SRP_B,
     SECRET_BLOCK, USER_ID_FOR_SRP.
  2. process_challenge()         → compute TIMESTAMP + PASSWORD_CLAIM_SIGNATURE.
  3. RespondToAuthChallenge PASSWORD_VERIFIER → receive the tokens.

Math follows AWS's amazon-cognito-identity-js / pycognito exactly (same N, g,
HKDF, and the non-zero-padded UTC timestamp format the signature is computed
over). NOTE: written from the spec and self-tested for internal consistency,
but NOT yet run against live Cognito — verify on first real login.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
from datetime import datetime, timezone

# RFC 3526 3072-bit MODP group (group 15) — the prime AWS Cognito uses.
# Grouped as the RFC prints it (48 hex per line) to keep transcription exact.
_N_HEX = (
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
    "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
    "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
    "DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64"
    "ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7"
    "ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6B"
    "F12FFA06D98A0864D87602733EC86A64521F2B18177B200C"
    "BBE117577A615D6C770988C0BAD946E208E24FA074E5AB31"
    "43DB5BFCE0FD108E4B82D120A93AD2CAFFFFFFFFFFFFFFFF"
)
_N = int(_N_HEX, 16)
assert _N.bit_length() == 3072, f"bad N: {_N.bit_length()} bits"
assert len(_N_HEX) == 768
_G = 2
_INFO_BITS = b"Caldera Derived Key"

_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ----------------------------------------------------------------------
# Low-level helpers
# ----------------------------------------------------------------------

def _hash_sha256(buf: bytes) -> str:
    """SHA-256 hex digest, lower-case, zero-padded to 64 chars."""
    return hashlib.sha256(buf).hexdigest().zfill(64)


def _hex_hash(hex_string: str) -> str:
    """SHA-256 of the *bytes* represented by a hex string."""
    return _hash_sha256(bytes.fromhex(hex_string))


def _pad_hex(value: int | str) -> str:
    """Cognito's padding: even length, and prepend '00' if the top bit is set."""
    if isinstance(value, int):
        hash_str = format(value, "x")
    else:
        hash_str = value
    if len(hash_str) % 2 == 1:
        hash_str = "0" + hash_str
    elif hash_str[0] in "89abcdefABCDEF":
        hash_str = "00" + hash_str
    return hash_str


def _get_random(nbytes: int) -> int:
    return int(binascii.hexlify(os.urandom(nbytes)), 16)


def _calculate_u(big_a: int, big_b: int) -> int:
    return int(_hex_hash(_pad_hex(big_a) + _pad_hex(big_b)), 16)


def _compute_hkdf(ikm: bytes, salt: bytes) -> bytes:
    """HKDF-SHA256 as Cognito uses it: 16-byte key, info = 'Caldera Derived Key'\\x01."""
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    info_bits = _INFO_BITS + b"\x01"
    return hmac.new(prk, info_bits, hashlib.sha256).digest()[:16]


# k = H(N | PAD(g)) — fixed for the group, computed once.
_K = int(_hex_hash(_pad_hex(_N) + _pad_hex(_G)), 16)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def generate_a() -> tuple[int, str]:
    """Return (small_a, SRP_A hex) with A = g^a mod N, A % N != 0."""
    while True:
        a = _get_random(128) % _N
        big_a = pow(_G, a, _N)
        if big_a % _N != 0:
            return a, format(big_a, "x")


def _format_timestamp() -> str:
    """UTC timestamp in the exact format the signature is computed over.

    e.g. 'Tue Oct 3 07:05:09 UTC 2026' — weekday/month are English abbrevs,
    the day is NOT zero-padded (matches amazon-cognito-identity-js), the time
    IS zero-padded.
    """
    now = datetime.now(timezone.utc)
    return (
        f"{_WEEKDAYS[now.weekday()]} {_MONTHS[now.month - 1]} {now.day} "
        f"{now:%H:%M:%S} UTC {now.year}"
    )


def process_challenge(
    *,
    small_a: int,
    big_a_hex: str,
    user_id_for_srp: str,
    password: str,
    user_pool_id: str,
    salt_hex: str,
    srp_b_hex: str,
    secret_block_b64: str,
) -> tuple[str, str]:
    """Compute (TIMESTAMP, PASSWORD_CLAIM_SIGNATURE) for the PASSWORD_VERIFIER step.

    user_id_for_srp is the USER_ID_FOR_SRP from the challenge (NOT the email).
    """
    pool_short = user_pool_id.split("_", 1)[1]
    big_a = int(big_a_hex, 16)
    big_b = int(srp_b_hex, 16)
    if big_b % _N == 0:
        raise ValueError("Cognito SRP: B mod N == 0 (bad server value)")

    u_value = _calculate_u(big_a, big_b)
    if u_value == 0:
        raise ValueError("Cognito SRP: u == 0 (bad A/B)")

    full_password = f"{pool_short}{user_id_for_srp}:{password}"
    full_password_hash = _hash_sha256(full_password.encode("utf-8"))
    x_value = int(_hex_hash(_pad_hex(salt_hex) + full_password_hash), 16)

    g_mod_pow_xn = pow(_G, x_value, _N)
    int_value2 = big_b - _K * g_mod_pow_xn
    s_value = pow(int_value2, small_a + u_value * x_value, _N)

    hkdf = _compute_hkdf(
        bytes.fromhex(_pad_hex(s_value)),
        bytes.fromhex(_pad_hex(u_value)),
    )

    timestamp = _format_timestamp()
    secret_block_bytes = base64.b64decode(secret_block_b64)
    msg = (
        pool_short.encode("utf-8")
        + user_id_for_srp.encode("utf-8")
        + secret_block_bytes
        + timestamp.encode("utf-8")
    )
    signature = base64.b64encode(
        hmac.new(hkdf, msg, hashlib.sha256).digest()
    ).decode("utf-8")

    return timestamp, signature
