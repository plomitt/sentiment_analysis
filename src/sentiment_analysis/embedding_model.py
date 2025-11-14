from sentence_transformers import SentenceTransformer

from sentiment_analysis.logging_utils import setup_logging

logger = setup_logging(__name__)

# Initialize model once for the entire module
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_MAX_TOKENS = EMBEDDING_MODEL.max_seq_length
EMBEDDING_TOKENIZER = EMBEDDING_MODEL._first_module().tokenizer
logger.debug("SentenceTransformer model initialized successfully")


def truncate_text_to_model_limit(text: str) -> str:
    """
    Truncate text to fit within the model's maximum token limit.

    Args:
        text: Input text to truncate.
        
    Returns:
        str: Truncated text fitting within model token limit.
    """
    # Tokenize with truncation
    tokens = EMBEDDING_TOKENIZER(
        text,
        truncation=True,
        max_length=EMBEDDING_MAX_TOKENS,
        return_tensors=None
    )

    # Convert back to string
    truncated_text = EMBEDDING_TOKENIZER.decode(
        tokens["input_ids"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    return truncated_text


__all__ = [
    "EMBEDDING_MODEL",
    "EMBEDDING_MAX_TOKENS",
    "EMBEDDING_TOKENIZER",
    "truncate_text_to_model_limit",
]