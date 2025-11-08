"""
Prompt management for Bitcoin news sentiment analysis.

This module contains specialized prompts and utility functions for
trading-focused sentiment scoring of Bitcoin news articles.
"""

MISSING_TITLE_TEXT = "No title available"
MISSING_BODY_TEXT = "No body content available"

BASE_PROMPT = """
    You are an expert Bitcoin trading analyst with deep understanding of cryptocurrency markets, technical analysis, and market sentiment indicators. Your task is to analyze Bitcoin news articles and provide sentiment scores from a trading perspective.

    SCORING FRAMEWORK:
    - 1.0-3.0: STRONG SELL - Major negative catalysts (regulatory crackdowns, security breaches, institutional abandonment, major adoption setbacks)
    - 3.1-5.0: WEAK SELL - Moderate negative factors (negative price action, minor regulatory concerns, reduced adoption metrics)
    - 5.1-6.0: NEUTRAL/HOLD - Mixed signals or routine market movements without clear directional bias
    - 6.1-8.0: WEAK BUY - Positive indicators (favorable developments, institutional interest, technical bullishness)
    - 8.1-10.0: STRONG BUY - Major positive catalysts (major institutional adoption, favorable regulation, breakthrough technical developments)

    ANALYSIS FACTORS:
    1. Market Impact: How will this affect Bitcoin price and trading volume?
    2. Regulatory Environment: Positive or negative regulatory implications
    3. Institutional Adoption: Signs of increased or decreased institutional interest
    4. Technical Indicators: Price levels, support/resistance, market structure
    5. Market Sentiment: Fear/greed indicators, social media sentiment
    6. Adoption Metrics: Real-world usage, merchant acceptance, network effects

    EXAMPLES:

    Example 1:
    Title: "SEC Approves Spot Bitcoin ETF for Major Investment Firm"
    Body: "The Securities and Exchange Commission has granted approval for a major investment firm to launch a spot Bitcoin ETF, opening the door for billions in institutional capital. The ETF will track Bitcoin's price directly and be available to retail investors through traditional brokerage accounts."
    Analysis: This is extremely bullish - regulatory approval + institutional access + capital inflow potential. Removes key regulatory barrier and creates easy institutional entry point.
    Score: 9.2
    Reasoning: SEC approval eliminates major regulatory uncertainty and creates massive institutional inflow channel. ETF structure makes Bitcoin accessible to traditional investors, likely driving significant demand and price appreciation.

    Example 2:
    Title: "Major Exchange Hacked, $500M in Bitcoin Stolen"
    Body: "Cryptocurrency exchange XYZ suffered a security breach resulting in the theft of $500 million worth of Bitcoin. The exchange has halted all withdrawals and trading. Hackers exploited a vulnerability in the exchange's hot wallet system."
    Analysis: Extremely negative for market sentiment - security concerns + exchange insolvency risk + potential market sell-off. Damages trust in crypto infrastructure.
    Score: 1.5
    Reasoning: Major security breach undermines confidence in crypto infrastructure and may trigger panic selling. Exchange solvency concerns could create systemic risk. Likely to cause immediate negative price action.

    Example 3:
    Title: "Tech Giant IntegraSoft Announces Bitcoin Integration for Cloud Services"
    Body: "IntegraSoft, the leading cloud infrastructure provider, announced today that customers can now pay for enterprise cloud services using Bitcoin. The integration will initially support 15 major enterprise clients with plans to expand to all customers by Q4 2025."
    Analysis: Positive institutional adoption news with credible enterprise integration. Large tech company endorsement adds legitimacy and creates new use case for Bitcoin in B2B payments.
    Score: 7.5
    Reasoning: Major tech company's Bitcoin integration for enterprise services represents significant institutional validation and creates new demand channel. Enterprise adoption trends are constructive for long-term value, though immediate market impact may be moderate.

    Example 4:
    Title: "Bitcoin Forms Golden Cross on Daily Chart, Traders Eye $125K Target"
    Body: "Technical analysts note that Bitcoin's 50-day moving average has crossed above its 200-day moving average, forming a classic golden cross pattern. Trading volume has been steadily increasing over the past week, supporting the bullish technical formation."
    Analysis: Strong bullish technical indicator with golden cross formation. However, technical patterns alone may not sustain upward movement without fundamental catalysts.
    Score: 6.8
    Reasoning: Golden cross is historically significant bullish indicator, especially when supported by increasing volume. However, technical patterns need fundamental confirmation for sustained price appreciation. Market structure favors bulls but requires catalyst validation.

    Example 5:
    Title: "Federal Reserve Chairman Signals Cautious Stance on Digital Asset Regulation"
    Body: "In a press conference today, the Federal Reserve Chairman emphasized the need for careful consideration of digital asset regulations, noting that while innovation is important, consumer protection and financial stability remain top priorities. Markets reacted negatively to the cautious tone."
    Analysis: Regulatory uncertainty from top monetary authority creates negative sentiment. However, cautious approach is expected and doesn't indicate immediate restrictive actions.
    Score: 4.7
    Reasoning: Cautious regulatory stance from Fed Chairman increases uncertainty about future regulatory environment. While not explicitly negative, the lack of clear support creates near-term headwinds for market sentiment and institutional adoption timelines.
"""


def get_system_prompt() -> str:
    return "You are an expert Bitcoin trading analyst."


def normalize_text(text: str, fallback_text: str) -> str:
    """
    Helper function to normalize text by handling None, empty strings, and missing values.

    Args:
        text: The text to normalize
        fallback_text: The text to use if original is None, empty, or missing

    Returns:
        Normalized text or fallback text
    """
    if text is None or text == '':
        return fallback_text
    return text


def get_sentiment_analysis_prompt_with_context(title: str, body: str, similar_articles: list = None, use_reasoning: bool = True) -> str:
    """
    Get the main sentiment analysis system prompt for Bitcoin news.

    This prompt uses few-shot learning with chain-of-thought reasoning to analyze
    Bitcoin news articles from a trading perspective, scoring them on a 1-10 scale
    where 1 suggests 'sell' and 10 suggests 'buy'.

    Args:
        title: The title of the article to analyze
        body: The body/content of the article to analyze
        similar_articles: Optional list of similar articles with their sentiment scores
            for reference in maintaining scoring consistency
        use_reasoning: Whether to include reasoning in the analysis (default: True)

    Returns:
        The complete sentiment analysis prompt with examples and scoring framework.
    """

    # Add similarity-based scoring instructions if similar articles are provided
    similar_articles_section = ""
    if similar_articles:
        similar_articles_section = """
        SIMILARITY-BASED SCORING REFERENCE:
        The following articles have been identified as similar to the current article and have been previously analyzed. Use these as reference points to maintain scoring consistency while still conducting independent analysis of the current article.
        """

        for i, article in enumerate(similar_articles, 1):
            article_title = normalize_text(article.get('title'), MISSING_TITLE_TEXT)
            article_body = normalize_text(article.get('body'), MISSING_BODY_TEXT)
            sentiment_score = article.get('sentiment_score', 'N/A')

            similar_articles_section += f"""
                Similar Article {i}:
                Title: {article_title}
                Body: {article_body}
                Sentiment Score: {sentiment_score}
            """

        similar_articles_section += """
        Consider these reference scores to maintain consistency in your analysis, but score the current article independently based on its own merits and market impact.
        """

    similar_articles_section += f"""
        Now analyze the given Bitcoin news article and provide your sentiment score and reasoning:

        Title: {normalize_text(title, MISSING_TITLE_TEXT)}
        Body: {normalize_text(body, MISSING_BODY_TEXT)}

        Provide your analysis in the following format:
        Success: [Whether the analysis was successful, bool]
        Score: [Your score from 1.0 to 10.0]"""

    if use_reasoning:
        similar_articles_section += """
        Reasoning: [Your concise reasoning focusing on trading implications and market impact]
        """


    return BASE_PROMPT + similar_articles_section


# Define the public API for this module
__all__ = [
    "get_system_prompt",
    "get_sentiment_analysis_prompt_with_context",
]
