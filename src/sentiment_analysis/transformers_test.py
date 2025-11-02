from sentence_transformers import SentenceTransformer
import time

sentences = [
    "Strategy Analysts Sound the Alarm on Saylor’s Bitcoin Premium - Bloomberg.com",
    "Elon Musk raises eyebrows with unexpected statement about bitcoin: 'It is impossible to fake' - Yahoo Finance",
    "Bitcoin and ether cap October with third weekly loss in the past four: CNBC Crypto World - CNBC",
    "Bitcoin ATMs enable cryptocurrency scams, federal prosecutor alleges - CBS News",
    "California Regulator Fines Bitcoin ATM Operator Coinhub $675K for Violating Law - Yahoo Finance",
    "California regulators levy stiff penalties against Bitcoin ATM operators - KTLA",
    "Bitcoin Is Sliding Amid Rate Cut Uncertainty. Here's Why 'Uptober' Never Happened. - Investopedia",
    "The Bitcoin White Paper Offered a Blueprint for a More Reliable Financial System - CoinDesk",
    "Bitcoin breaks October streak with first monthly loss since 2018 - Reuters",
    "Bitcoin Price, Ethereum Drop. How Powell Took Down the Crypto Rally. - Barron's",
]

model = SentenceTransformer("all-MiniLM-L6-v2")

start_time = time.perf_counter()

embeddings = model.encode(sentences)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")



def get_embedded_articles(analyzed_articles: List[Dict[str, Any]], batch_size: Optional[int] = 32) -> List[Dict[str, Any]]:
    """
    Generate embeddings for a batch of analyzed articles using batch processing.

    Args:
        analyzed_articles: List of analyzed article dictionaries
        batch_size: Number of articles to process in each batch (default: 32)

    Returns:
        List of analyzed articles with embedding vectors added

    Raises:
        ValueError: If no articles provided
        Exception: If embedding generation fails
    """
    if not analyzed_articles:
        logger.warning("No articles provided for embedding generation")
        return []

    start_time = time.time()
    logger.info(f"Generating embeddings for {len(analyzed_articles)} articles with batch size {batch_size}")

    try:
        # Initialize model once for the entire batch
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.debug("SentenceTransformer model initialized successfully")

        embedded_articles = []

        # Process articles in batches to manage memory efficiently
        for i in range(0, len(analyzed_articles), batch_size):
            batch = analyzed_articles[i:i + batch_size]
            batch_start_time = time.time()

            # Prepare batch texts for embedding
            batch_texts = [make_embedding_text(article) for article in batch]

            # Generate embeddings for the entire batch at once
            batch_embeddings = model.encode(batch_texts, batch_size=len(batch))

            # Create embedded articles by adding embeddings to original articles
            for article, embedding in zip(batch, batch_embeddings):
                embedded_article = dict(article)  # Create a copy to avoid modifying original
                embedded_article["embedding"] = embedding.tolist()
                embedded_articles.append(embedded_article)

            batch_duration = time.time() - batch_start_time
            logger.debug(f"Processed batch {i//batch_size + 1}: {len(batch)} articles in {batch_duration:.2f}s")

        total_duration = time.time() - start_time
        avg_time_per_article = total_duration / len(analyzed_articles)

        logger.info(f"Successfully generated embeddings for {len(embedded_articles)} articles in {total_duration:.2f}s (avg: {avg_time_per_article:.3f}s per article).")

        return embedded_articles

    except Exception as e:
        logger.error(f"Failed to generate embeddings for articles: {e}")
        raise