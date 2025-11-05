from pprint import pprint
from sentence_transformers import SentenceTransformer
import psycopg
from pgvector.psycopg import register_vector
from typing import Dict, List, Any, Optional, Set

from sentiment_analysis.db_utils import get_postgres_connection_string
from sentiment_analysis.pipeline import make_embedding_text


model = SentenceTransformer("all-MiniLM-L6-v2")

article = {
    'title': 'Strategy Analysts Sound the Alarm on Saylor’s Bitcoin Premium - Bloomberg.com',
    'body': '4 hours ago - At least three analysts — from Cantor Fitzgerald LP, TD Cowen and Maxim Group LLC — lowered their price target on the company following its earnings on Friday, taking the average target to its lowest since May.',
}

embed_text = make_embedding_text(article)
embedding = model.encode(embed_text).tolist()


def fetch_similar_articles(conn: psycopg.Connection, embedding: List, limit: int = 5) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT title, body, sentiment_score
            FROM articles
            ORDER BY embedding <=> %s::vector(384)
            LIMIT %s
            """,
            (embedding, limit)
        )
        rows = cur.fetchall()
        similar_articles = [{"title": r[0], "body": r[1], "sentiment_score": r[2]} for r in rows]
    
    return similar_articles

conn_string = get_postgres_connection_string()
with psycopg.connect(conn_string, autocommit=True) as conn:
    register_vector(conn)
    similar_articles = fetch_similar_articles(conn, embedding, limit=5)
    pprint(similar_articles)