import json

import numpy as np

from llm_fingerprinter.fingerprint_store import FingerprintStore


def test_save_fingerprint_persists_response_metadata(tmp_path):
    store = FingerprintStore(tmp_path)
    fingerprint = {
        "vector": np.array([1.0, 2.0], dtype=np.float32),
        "raw_features": {},
        "metadata": {"feature_dim": 2},
        "responses_sample": [
            {
                "prompt": "hello",
                "response": "world",
                "response_metadata": {
                    "provider": "test-provider",
                    "model": "model-a",
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": np.int64(3),
                        "output_tokens": np.int64(4),
                        "total_tokens": np.int64(7),
                    },
                    "raw": {
                        "utf8_bytes": b"hello",
                        "binary_bytes": b"\xff\x00",
                    },
                },
                "feature_metadata": {
                    "embedding_dim": np.int64(384),
                    "empty_response": np.bool_(False),
                },
            }
        ],
    }

    path = store.save_fingerprint(fingerprint, "model-a", family="gpt")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["responses_sample"] == [
        {
            "prompt": "hello",
            "response": "world",
            "response_metadata": {
                "provider": "test-provider",
                "model": "model-a",
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 4,
                    "total_tokens": 7,
                },
                "raw": {
                    "utf8_bytes": {
                        "__type__": "bytes",
                        "base64": "aGVsbG8=",
                    },
                    "binary_bytes": {
                        "__type__": "bytes",
                        "base64": "/wA=",
                    },
                },
            },
            "feature_metadata": {
                "embedding_dim": 384,
                "empty_response": False,
            },
        }
    ]
