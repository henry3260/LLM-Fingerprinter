from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from llm_fingerprinter.contracts.feature import Feature, FeatureVector

logger = logging.getLogger(__name__)

def _setup_nltk():
    """Ensure NLTK data packages are present, downloading only if missing.

    Data is stored in llm_fingerprinter/nltk_data/ — next to this file —
    keeping everything self-contained within the project directory.
    """
    import nltk
    from pathlib import Path

    # Store NLTK data alongside this file: llm_fingerprinter/nltk_data/
    nltk_data_dir = str(Path(__file__).parent / "nltk_data")

    # Prepend so it's checked first, before any system/home locations
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)

    packages_to_try = ['punkt_tab', 'punkt', 'stopwords']
    for package in packages_to_try:
        try:
            # Check if already downloaded — avoids a network call on every import
            nltk.data.find(f"tokenizers/{package}" if package != 'stopwords'
                           else "corpora/stopwords")
        except LookupError:
            try:
                nltk.download(package, download_dir=nltk_data_dir, quiet=True)
            except Exception as e:
                logger.debug(f"NLTK {package} download failed: {e}")

    try:
        from nltk.tokenize import sent_tokenize, word_tokenize
        sent_tokenize("Test sentence.")
        word_tokenize("Test words.")
    except Exception as e:
        logger.warning(f"NLTK tokenizer unavailable, will use fallback: {e}")

_setup_nltk()

SentenceTransformer = None


def _get_sentence_transformer_class():
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as loaded_transformer

        SentenceTransformer = loaded_transformer
    return SentenceTransformer


# Safe NLTK imports with fallbacks
try:
    from nltk import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    logger.warning("NLTK not available, using basic tokenization")


def _safe_sent_tokenize(text):
    """Tokenize text into sentences with fallback."""
    if _NLTK_AVAILABLE:
        try:
            return sent_tokenize(text)
        except Exception:
            pass
    # Fallback: split on sentence-ending punctuation
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _safe_word_tokenize(text: str):
    """Tokenize text into words with fallback."""
    if _NLTK_AVAILABLE:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    # Fallback: simple split with punctuation handling
    import re
    return re.findall(r'\b\w+\b', text.lower())


def _get_stopwords():
    """Get English stopwords with fallback."""
    if _NLTK_AVAILABLE:
        try:
            return set(stopwords.words('english'))
        except Exception:
            pass
    # Fallback: basic English stopwords
    return {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'and', 'but', 'if', 'or', 'because', 'until', 'while', 'this',
            'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our',
            'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'he', 'him',
            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
            'which', 'who', 'whom'}


class FeatureExtractor:
    
    LINGUISTIC_DIM = 12
    BEHAVIORAL_DIM = 6
    FEATURE_NAMESPACE = "llm_response"
    FEATURE_SCHEMA_VERSION = "feature_extractor.v1"
    LINGUISTIC_FEATURE_NAMES = (
        "total_chars",
        "total_words",
        "type_token_ratio",
        "avg_word_length",
        "sentence_count",
        "avg_sentence_length",
        "punctuation_ratio",
        "code_block_ratio",
        "structural_marker_ratio",
        "token_entropy",
        "ai_marker_count",
        "capital_ratio",
    )
    BEHAVIORAL_FEATURE_NAMES = (
        "refusal_score",
        "format_adherence_score",
        "reasoning_presence_score",
        "instruction_compliance_score",
        "length_normalization_score",
        "formality_score",
    )
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize feature extractor.
        
        Args:
            model_name: SentenceTransformer model name. 
                       Note: all-MiniLM-L6-v2 outputs 384-dim embeddings.
        """
        self.model_name = model_name
        transformer_cls = _get_sentence_transformer_class()
        self.embedding_model = transformer_cls(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.stop_words = _get_stopwords()
            
        logger.info(f"Initialized FeatureExtractor with {model_name} "
                   f"(embedding dim: {self.embedding_dim})")
    
    def get_feature_names(self) -> list[str]:
        """Return the stable feature order used by vector serialization."""

        return (
            [f"embedding_{i}" for i in range(self.embedding_dim)]
            + list(self.LINGUISTIC_FEATURE_NAMES)
            + list(self.BEHAVIORAL_FEATURE_NAMES)
        )

    @staticmethod
    def feature_vector_to_array(feature_vector: FeatureVector) -> np.ndarray:
        """Convert a FeatureVector to the legacy numpy representation."""

        return np.array(
            [float(item.value) for item in feature_vector.items],
            dtype=np.float32,
        )

    def _empty_feature_array(self) -> np.ndarray:
        return np.zeros(self.get_feature_dim(), dtype=np.float32)

    def _coerce_response_text(self, response: Any) -> str:
        text = getattr(response, "text", response)
        if text is None:
            return ""
        return str(text)

    def _build_feature_vector(
        self,
        prompt: str,
        response: str,
        values: np.ndarray,
        empty_response: bool = False,
    ) -> FeatureVector:
        names = self.get_feature_names()
        values = values.astype(np.float32).ravel()
        if len(values) != len(names):
            raise ValueError(
                f"Feature dimension mismatch: {len(values)} values for {len(names)} names"
            )

        return FeatureVector(
            items=[
                Feature(name=name, value=float(value))
                for name, value in zip(names, values)
            ],
            namespace=self.FEATURE_NAMESPACE,
            metadata={
                "schema_version": self.FEATURE_SCHEMA_VERSION,
                "embedding_model": self.model_name,
                "embedding_dim": int(self.embedding_dim),
                "linguistic_dim": self.LINGUISTIC_DIM,
                "behavioral_dim": self.BEHAVIORAL_DIM,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "empty_response": empty_response,
            },
        )

    def extract_batch_vectors(self, prompt_response_pairs: list) -> list[FeatureVector]:
        """Extract structured FeatureVector objects for prompt/response pairs.

        Batches the embedding step (the expensive GPU/CPU forward pass) so that
        N responses are encoded in one call instead of N separate calls.
        Linguistic and behavioral features are still computed per-response
        (they are pure Python and not the bottleneck).

        Args:
            prompt_response_pairs: list of (prompt_str, response_str) tuples

        Returns:
            List of FeatureVector, one per pair, in the same order.
            Zero-vectors are returned for empty/failed responses.
        """
        if not prompt_response_pairs:
            return []

        normalized_pairs = [
            (str(prompt), self._coerce_response_text(response))
            for prompt, response in prompt_response_pairs
        ]
        responses = [response for _, response in normalized_pairs]

        try:
            embeddings = self.embedding_model.encode(
                responses,
                batch_size=32,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
        except Exception as e:
            logger.error(f"Batch embedding failed, falling back per-response: {e}")
            embeddings = np.stack([
                self._embedding_features(r) for r in responses
            ])

        results = []
        for (prompt, response), embedding in zip(normalized_pairs, embeddings):
            if not response or not response.strip():
                results.append(
                    self._build_feature_vector(
                        prompt,
                        response,
                        self._empty_feature_array(),
                        empty_response=True,
                    )
                )
                continue
            ling = self._linguistic_features(response)
            beh = self._behavioral_features(prompt, response)
            results.append(
                self._build_feature_vector(
                    prompt,
                    response,
                    np.concatenate([embedding.astype(np.float32), ling, beh]),
                )
            )

        return results

    def extract_batch(self, prompt_response_pairs: list) -> list:
        """Extract legacy numpy arrays for multiple prompt/response pairs."""

        return [
            self.feature_vector_to_array(feature_vector)
            for feature_vector in self.extract_batch_vectors(prompt_response_pairs)
        ]

    def extract_vector(self, prompt: str, response: Any) -> FeatureVector:
        """Extract a structured feature vector for one prompt/response pair."""

        response_text = self._coerce_response_text(response)
        if not response_text or not response_text.strip():
            logger.warning("Empty response, returning zero features")
            return self._build_feature_vector(
                prompt,
                response_text,
                self._empty_feature_array(),
                empty_response=True,
            )
        
        embedding_features = self._embedding_features(response_text)
        linguistic_features = self._linguistic_features(response_text)
        behavioral_features = self._behavioral_features(prompt, response_text)
        
        all_features = np.concatenate([
            embedding_features,
            linguistic_features,
            behavioral_features
        ])
        
        return self._build_feature_vector(prompt, response_text, all_features)

    def extract(self, prompt: str, response: Any):
        """Extract the legacy numpy array representation for one response."""

        return self.feature_vector_to_array(self.extract_vector(prompt, response))
    
    def get_feature_dim(self):

        return self.embedding_dim + self.LINGUISTIC_DIM + self.BEHAVIORAL_DIM
    
    def _embedding_features(self, response: str):
        try:
            embedding = self.embedding_model.encode(response, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _linguistic_features(self, response: str):
        """
        Extract linguistic features (12-dim).
        
        Features:
            0: Total characters
            1: Total words
            2: Type-token ratio (vocabulary diversity)
            3: Average word length
            4: Number of sentences
            5: Average sentence length
            6: Punctuation ratio
            7: Code block ratio
            8: Structural markers ratio
            9: Token entropy
            10: AI marker count
            11: Capital letter ratio
        """
        features = []
        
        # Length features
        features.append(len(response))  # total chars
        
        # Tokenize safely using our fallback-enabled functions
        words = _safe_word_tokenize(response.lower())
        
        features.append(len(words))  # total words
        
        # Vocabulary features
        unique_words = len(set(words))
        type_token_ratio = unique_words / max(len(words), 1)
        features.append(type_token_ratio)
        
        # Complexity
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        features.append(avg_word_len)
        
        # Sentence stats using safe tokenizer
        sentences = _safe_sent_tokenize(response)
        num_sentences = max(len(sentences), 1)
        
        avg_sent_len = len(words) / max(num_sentences, 1)
        features.append(num_sentences)
        features.append(avg_sent_len)
        
        # Punctuation ratio
        punctuation_count = sum(1 for c in response if c in '.,!?;:')
        features.append(punctuation_count / max(len(response), 1))
        
        # Code blocks (using ``` markers)
        code_block_count = response.count("```")
        code_ratio = code_block_count / max(len(response), 1) * 100  # Scale up for visibility
        features.append(code_ratio)
        
        # Structural markers (bullet points, numbered lists)
        bullet_patterns = [
            r'^[-*•]\s',  # Bullet points (-, *, •)
            r'^\d+\.\s',  # Numbered lists (1., 2., etc.)
        ]
        struct_count = 0
        for line in response.split('\n'):
            line = line.strip()
            for pattern in bullet_patterns:
                if re.match(pattern, line):
                    struct_count += 1
                    break
        features.append(struct_count / max(len(response.split('\n')), 1))
        
        # Token entropy
        if words:
            word_freq = {}
            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1
            probs = np.array(list(word_freq.values())) / len(words)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
        else:
            entropy = 0
        features.append(entropy)
        
        # Characteristic AI phrases
        ai_markers = sum([
            response.lower().count("as an ai"),
            response.lower().count("as a language model"),
            response.lower().count("as an artificial"),
            response.lower().count("i cannot"),
            response.lower().count("i can't"),
            response.lower().count("i'm not able"),
            response.lower().count("i am not able"),
        ])
        features.append(ai_markers)
        
        # Capital ratio
        capital_count = sum(1 for c in response if c.isupper())
        capital_ratio = capital_count / max(len(response), 1)
        features.append(capital_ratio)
        
        return np.array(features, dtype=np.float32)
    
    def _behavioral_features(self, prompt: str, response: str):
        """
        Extract behavioral features (6-dim).
        
        Features:
            0: Refusal score
            1: Format adherence score
            2: Reasoning presence score
            3: Instruction compliance score
            4: Length normalization score
            5: Formality score
        """
        features = []
        
        response_lower = response.lower()
        prompt_lower = prompt.lower()
        word_count = max(len(response.split()), 1)
        
        refusal_keywords = [
            "cannot", "can't", "unable", "not able", "refuse", 
            "apologize", "sorry", "won't", "will not", "inappropriate"
        ]
        refusal_score = sum(1 for kw in refusal_keywords if kw in response_lower) / word_count
        features.append(refusal_score)
        
        format_markers = (
            response.count("1.") + response.count("2.") + response.count("3.") +
            response.count("- ") + response.count("* ") +
            response.count("**") + response.count("##")
        )
        format_score = min(format_markers / word_count, 1.0)
        features.append(format_score)
        
        reasoning_keywords = [
            "therefore", "step", "reason", "because", "thus",
            "first", "second", "third", "finally", "however",
            "consequently", "as a result", "let me", "let's"
        ]
        reasoning_score = sum(1 for kw in reasoning_keywords if kw in response_lower) / word_count
        features.append(reasoning_score)
        
        compliance = 0.5
        
        if "list" in prompt_lower or "enumerate" in prompt_lower:
            # Check for list structure
            has_list = (
                response.count("\n") > 2 and 
                (response.count("- ") > 0 or response.count("1.") > 0 or response.count("* ") > 0)
            )
            compliance = 1.0 if has_list else 0.0
        elif "code" in prompt_lower or "implement" in prompt_lower or "write a function" in prompt_lower:
            # Check for code block
            has_code = "```" in response or response.count("def ") > 0 or response.count("function") > 0
            compliance = 1.0 if has_code else 0.0
        elif "brief" in prompt_lower or "short" in prompt_lower:
            # Check for brevity
            compliance = 1.0 if word_count < 100 else 0.5 if word_count < 200 else 0.0
        elif "detailed" in prompt_lower or "explain" in prompt_lower:
            # Check for detail
            compliance = 1.0 if word_count > 100 else 0.5 if word_count > 50 else 0.0
            
        features.append(compliance)
        
        length_score = 1.0 - min(abs(word_count - 150) / 300, 1.0)
        features.append(length_score)
        
        formal_words = [
            "indeed", "furthermore", "thus", "moreover", "however",
            "nevertheless", "consequently", "therefore", "accordingly",
            "hence", "subsequently", "nonetheless"
        ]
        formal_score = sum(1 for w in formal_words if w in response_lower) / word_count
        features.append(formal_score)
        
        return np.array(features, dtype=np.float32)
