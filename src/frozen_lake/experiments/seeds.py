"""Stable seeds: every map, episode and simulation seed derives from the master seed."""

import hashlib

# Version of the seed derivation below; recorded with every run.
SEED_SCHEME = "v1"


def derive_seed(master_seed: int, *parts: object) -> int:
    """A stable 63-bit seed for one role, from the master seed and a description of the role.

    Uses SHA-256 of a canonical string, never Python's per-process salted hash(), so the same
    description gives the same seed on every run and machine.
    """
    text = "/".join([SEED_SCHEME, str(master_seed), *map(str, parts)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") >> 1
