"""
Demo script to showcase the Bitcoin sentiment analyzer functionality.

This script demonstrates how to:
1. Analyze a single article
2. Load and analyze multiple articles from JSON
3. Display results in a user-friendly format
"""

import json
import os
from sentiment_analyzer import (
    analyze_article,
    analyze_news_file,
    load_articles_from_json,
    create_client
)

def demo_single_article():
    """Demonstrate analyzing a single article."""
    print("="*60)
    print("DEMO: Single Article Analysis")
    print("="*60)

    # Sample article
    sample_title = "Bitcoin Reaches New All-Time High as Institutional Adoption Accelerates"
    sample_body = """
    Bitcoin surged to a new all-time high above $120,000 today as major financial institutions announced significant Bitcoin investments.
    BlackRock revealed it has allocated $5 billion to Bitcoin across multiple funds, while Fidelity launched a new Bitcoin-focused product
    for retail investors. Market analysts attribute the rally to growing institutional acceptance and improved regulatory clarity.
    Technical indicators suggest further upside potential, with on-chain metrics showing strong holder confidence.
    """

    # Analyze sentiment
    print(f"Analyzing: {sample_title}")
    print("-" * 60)

    sentiment = analyze_article(sample_title, sample_body)

    print(f"📊 Sentiment Score: {sentiment.score}/10")
    print(f"💭 Reasoning: {sentiment.reasoning}")

    # Interpret the score
    if sentiment.score <= 3.0:
        signal = "🔴 STRONG SELL"
    elif sentiment.score <= 5.0:
        signal = "🟡 WEAK SELL"
    elif sentiment.score <= 6.0:
        signal = "⚪ NEUTRAL/HOLD"
    elif sentiment.score <= 8.0:
        signal = "🟢 WEAK BUY"
    else:
        signal = "🟢 STRONG BUY"

    print(f"🎯 Trading Signal: {signal}")
    print()

def demo_batch_analysis():
    """Demonstrate batch analysis of existing news."""
    print("="*60)
    print("DEMO: Batch Analysis of News Articles")
    print("="*60)

    # Get script directory for correct file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, 'news_with_sentiment.json')

    # Load existing results
    try:
        with open(results_file, 'r') as f:
            articles = json.load(f)

        print(f"Loaded {len(articles)} articles with sentiment analysis")
        print("-" * 60)

        # Show top 3 bullish articles
        print("🟢 TOP 3 BULLISH ARTICLES:")
        bullish_articles = sorted(articles, key=lambda x: x['sentiment']['score'], reverse=True)[:3]
        for i, article in enumerate(bullish_articles, 1):
            print(f"{i}. Score: {article['sentiment']['score']}/10")
            print(f"   Title: {article['title'][:80]}...")
            print(f"   Reasoning: {article['sentiment']['reasoning'][:100]}...")
            print()

        print("🔴 TOP 3 BEARISH ARTICLES:")
        bearish_articles = sorted(articles, key=lambda x: x['sentiment']['score'])[:3]
        for i, article in enumerate(bearish_articles, 1):
            print(f"{i}. Score: {article['sentiment']['score']}/10")
            print(f"   Title: {article['title'][:80]}...")
            print(f"   Reasoning: {article['sentiment']['reasoning'][:100]}...")
            print()

        # Overall sentiment
        scores = [article['sentiment']['score'] for article in articles]
        avg_score = sum(scores) / len(scores)
        print(f"📈 OVERALL SENTIMENT:")
        print(f"   Average Score: {avg_score:.2f}/10")
        print(f"   Articles Analyzed: {len(articles)}")

        if avg_score <= 5.0:
            overall_sentiment = "🔴 BEARISH"
        elif avg_score <= 6.0:
            overall_sentiment = "⚪ NEUTRAL"
        else:
            overall_sentiment = "🟢 BULLISH"

        print(f"   Market Bias: {overall_sentiment}")

    except FileNotFoundError:
        print("❌ Error: news_with_sentiment.json not found. Run sentiment_analyzer.py first.")
    except Exception as e:
        print(f"❌ Error loading results: {e}")

def demo_score_interpretation():
    """Show how to interpret different scores."""
    print("="*60)
    print("DEMO: Score Interpretation Guide")
    print("="*60)

    interpretations = {
        "1.0-3.0": {
            "signal": "🔴 STRONG SELL",
            "description": "Major negative catalysts",
            "examples": ["Regulatory crackdowns", "Security breaches", "Major institutional abandonment"]
        },
        "3.1-5.0": {
            "signal": "🟡 WEAK SELL",
            "description": "Moderate negative factors",
            "examples": ["Negative price action", "Minor regulatory concerns", "Reduced adoption metrics"]
        },
        "5.1-6.0": {
            "signal": "⚪ NEUTRAL/HOLD",
            "description": "Mixed signals or routine movements",
            "examples": ["Technical consolidation", "Mixed news flow", "No clear catalysts"]
        },
        "6.1-8.0": {
            "signal": "🟢 WEAK BUY",
            "description": "Positive indicators",
            "examples": ["Favorable developments", "Growing institutional interest", "Bullish technical patterns"]
        },
        "8.1-10.0": {
            "signal": "🟢 STRONG BUY",
            "description": "Major positive catalysts",
            "examples": ["Major institutional adoption", "Favorable regulation", "Breakthrough developments"]
        }
    }

    for score_range, info in interpretations.items():
        print(f"{score_range}: {info['signal']}")
        print(f"   Description: {info['description']}")
        print(f"   Examples: {', '.join(info['examples'])}")
        print()

def main():
    """Run all demos."""
    print("🚀 Bitcoin News Sentiment Analyzer - Demo")
    print("This demo showcases the sentiment analysis capabilities for Bitcoin trading.")
    print()

    # Run demos
    demo_single_article()
    demo_score_interpretation()
    demo_batch_analysis()

    print("="*60)
    print("✅ Demo Complete!")
    print("To analyze your own articles:")
    print("1. Run sentiment_analyzer.py for batch analysis")
    print("2. Or use analyze_article() function in your code")
    print("3. Or use analyze_news_file() for complete pipeline")
    print("="*60)

if __name__ == "__main__":
    main()