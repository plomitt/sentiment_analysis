"""
Prompt manager for Bitcoin news sentiment analysis.
Contains specialized prompts for trading-focused sentiment scoring.
"""

from typing import Dict, Any

def get_sentiment_analysis_prompt() -> str:
    """
    Get the main sentiment analysis system prompt for Bitcoin news.

    This prompt uses few-shot learning with chain-of-thought reasoning to analyze
    Bitcoin news articles from a trading perspective, scoring them on a 1-10 scale
    where 1 suggests 'sell' and 10 suggests 'buy'.
    """
    return """
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
        Title: "Square Enables First Bitcoin Payment at US Coffee Chain"
        Body: "Square has enabled Compass Coffee as the first business to accept Bitcoin payments through Square's point-of-sale terminal. All 10 wallet tests succeeded at the coffee shop instantly as Square prepares worldwide rollout."
        Analysis: Positive adoption news but relatively small scale. Real-world usage increasing, payment infrastructure improving. However, single merchant adoption has minimal market impact.
        Score: 6.8
        Reasoning: Real-world adoption milestone with successful payment processing, but limited immediate market impact. Shows improving payment infrastructure and merchant acceptance, supporting long-term bullish thesis.

        Example 4:
        Title: "Bitcoin Price Watch: Short-Term Charts Hint at Accumulation"
        Body: "Bitcoin may be clinging just above the $111K ledge, but the charts are less interested in a party and more in plotting their next move—quietly, methodically, and with just a hint of drama. Technical indicators suggest accumulation phase."
        Analysis: Technical analysis showing accumulation patterns. Generally constructive for price but lacks fundamental catalyst. Short-term bullish but need confirmation.
        Score: 6.2
        Reasoning: Technical indicators suggest accumulation and potential bullish setup, but lacks fundamental catalysts for significant price movement. Charts constructive but need volume confirmation for breakout.

        Example 5:
        Title: "Bitcoin Holds $110K, But Traders Just Bet $1.15B on Crash to $104K"
        Body: "Bitcoin institutional traders bet over $1.15 billion on downside protection in 24 hours with put options accounting for 28% of market transactions targeting $104,000 to $108,000 range."
        Analysis: Large institutional hedging activity suggests smart money expecting downside. Significant options flow for downside protection. However, price still holding key levels creates mixed picture.
        Score: 4.3
        Reasoning: Heavy institutional positioning for downside protection suggests smart money anticipating price decline. Despite current price resilience, $1.15B in downside bets indicates institutional concern about near-term price action.

        Now analyze the given Bitcoin news article and provide your sentiment score and reasoning:

        Title: {title}
        Body: {body}

        Provide your analysis in the following format:
        Score: [Your score from 1.0 to 10.0]
        Reasoning: [Your concise reasoning focusing on trading implications and market impact]
        """

def get_sentiment_analysis_prompt_with_context(title: str, body: str) -> str:
    """
    Get the sentiment analysis prompt with article context filled in.

    Args:
        title: Article title
        body: Article body content

    Returns:
        Formatted prompt with article context
    """
    base_prompt = get_sentiment_analysis_prompt()
    return base_prompt.format(title=title, body=body)

def get_sentiment_score_definitions() -> Dict[str, str]:
    """
    Get detailed definitions for each sentiment score range.
    Useful for reference and ensuring consistency in scoring.

    Returns:
        Dictionary mapping score ranges to their meanings
    """
    return {
        "1.0-3.0": "STRONG SELL - Major negative catalysts (regulatory crackdowns, security breaches, institutional abandonment)",
        "3.1-5.0": "WEAK SELL - Moderate negative factors (negative price action, minor regulatory concerns, reduced adoption)",
        "5.1-6.0": "NEUTRAL/HOLD - Mixed signals or routine market movements without clear directional bias",
        "6.1-8.0": "WEAK BUY - Positive indicators (favorable developments, institutional interest, technical bullishness)",
        "8.1-10.0": "STRONG BUY - Major positive catalysts (major institutional adoption, favorable regulation, breakthrough developments)"
    }

def validate_sentiment_score(score: float) -> bool:
    """
    Validate that a sentiment score is within the acceptable range.

    Args:
        score: The sentiment score to validate

    Returns:
        True if valid, False otherwise
    """
    return isinstance(score, (int, float)) and 1.0 <= score <= 10.0