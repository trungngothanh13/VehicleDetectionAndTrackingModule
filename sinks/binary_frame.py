import json
import struct
from typing import Any, Dict, Optional, Tuple


def pack_binary_frame(header: Dict[str, Any], payload: bytes) -> bytes:
    """Packs [4-byte uint32 BE headerLen][UTF-8 JSON header][raw JPEG bytes]."""
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(header_bytes)) + header_bytes + payload


def unpack_binary_frame(raw: bytes) -> Optional[Tuple[Dict[str, Any], bytes]]:
    """Unpacks [4-byte uint32 BE headerLen][UTF-8 JSON header][raw JPEG bytes]."""
    if len(raw) < 4:
        return None
    header_len = struct.unpack(">I", raw[:4])[0]
    if header_len <= 0 or 4 + header_len > len(raw):
        return None
    try:
        header = json.loads(raw[4:4 + header_len].decode("utf-8"))
        return header, raw[4 + header_len:]
    except Exception:
        return None
