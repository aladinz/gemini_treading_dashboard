"""
AI Helper module for generating trading insights using OpenAI
"""
import os
from openai import OpenAI

# API Configuration - Replace with your actual API key or use environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your_openai_api_key_here')

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_trading_insights(ticker, price, signal, confidence, technical_data):
    """
    Generate comprehensive AI-powered trading insights.
    
    Parameters:
    - ticker: Stock symbol
    - price: Current stock price
    - signal: Trading signal (BUY, SELL, HOLD, etc.)
    - confidence: Signal confidence percentage
    - technical_data: Dictionary containing technical indicators
        {
            'rsi': float,
            'macd': float,
            'macd_signal': float,
            'sma20': float,
            'sma50': float,
            'sma200': float,
            'support_levels': list,
            'resistance_levels': list,
            'entry_price': float,
            'stop_loss': float,
            'take_profit': float
        }
    """
    
    try:
        # Determine position type for clarity
        is_buy_signal = signal in ['BUY', 'STRONG BUY']
        is_sell_signal = signal in ['SELL', 'STRONG SELL']
        
        # Build appropriate setup description based on signal
        if is_buy_signal:
            setup_description = f"""**Long Position Setup (BUY):**
- Entry Point: ${technical_data.get('entry_price', 0):.2f}
- Stop Loss: ${technical_data.get('stop_loss', 0):.2f} (exit if price drops below)
- Take Profit Target: ${technical_data.get('take_profit', 0):.2f} (target price above entry)"""
        elif is_sell_signal:
            setup_description = f"""**Short Position Setup (SELL) or Exit Signal:**
- Current Price: ${price:.2f}
- Downside Target: ${technical_data.get('take_profit', 0):.2f} (expected price decline)
- Stop Loss: ${technical_data.get('stop_loss', 0):.2f} (exit if price rises above)
Note: This is a SELL/SHORT signal - the target price ${technical_data.get('take_profit', 0):.2f} is BELOW current price, which is correct for shorting or exiting long positions."""
        else:
            setup_description = f"""**Position Setup:**
- Current Price: ${price:.2f}
- Suggested Entry: ${technical_data.get('entry_price', 0):.2f}
- Stop Loss: ${technical_data.get('stop_loss', 0):.2f}
- Take Profit: ${technical_data.get('take_profit', 0):.2f}"""
        
        # Build comprehensive context for AI
        prompt = f"""You are an expert swing trading advisor. Analyze this stock setup and provide actionable insights.

**Stock:** {ticker}
**Current Price:** ${price:.2f}
**Signal:** {signal} (Confidence: {confidence}%)

**Technical Indicators:**
- RSI: {technical_data.get('rsi', 'N/A')}
- MACD: {technical_data.get('macd', 'N/A')}
- MACD Signal: {technical_data.get('macd_signal', 'N/A')}
- SMA 20: ${technical_data.get('sma20', 0):.2f}
- SMA 50: ${technical_data.get('sma50', 0):.2f}
- SMA 200: ${technical_data.get('sma200', 0):.2f}

**Support Levels:** {', '.join([f'${s:.2f}' for s in technical_data.get('support_levels', [])]) or 'None identified'}
**Resistance Levels:** {', '.join([f'${r:.2f}' for r in technical_data.get('resistance_levels', [])]) or 'None identified'}

{setup_description}

IMPORTANT CONTEXT:
- For BUY signals: This is a LONG position where you buy now and sell higher later. Take profit is ABOVE entry.
- For SELL signals: This can be a SHORT position (sell high, buy back low) OR exiting a long position. Target is BELOW current price.
- Make sure your Action Plan correctly reflects whether this is a long or short position.

Provide a structured analysis with these sections:

1. **Signal Analysis** (2-3 sentences): Why this signal was generated and its strength
2. **Key Factors** (3-4 bullet points): Most important technical factors to consider
3. **Risk Assessment** (2 sentences): What could go wrong and risk level
4. **Action Plan** (2-3 sentences): Clear next steps - be specific about whether to ENTER a position, EXIT a position, or WAIT. For SELL signals, clarify if this is for shorting or exiting longs.
5. **Learning Tip** (1-2 sentences): Educational insight about the technical setup

Keep it concise, actionable, and educational. Use emojis sparingly for clarity."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional swing trading advisor providing clear, actionable insights. Be concise but thorough."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ Unable to generate AI insights: {str(e)}\n\nPlease check your API configuration."


def generate_journal_insights(trades_summary):
    """
    Generate insights from trading journal history.
    
    Parameters:
    - trades_summary: Dictionary containing trading statistics
        {
            'total_trades': int,
            'win_rate': float,
            'profit_loss': float,
            'avg_win': float,
            'avg_loss': float,
            'best_win': float,
            'worst_loss': float
        }
    """
    
    try:
        prompt = f"""Analyze this trader's performance and provide constructive feedback.

**Trading Statistics:**
- Total Trades: {trades_summary.get('total_trades', 0)}
- Win Rate: {trades_summary.get('win_rate', 0):.1f}%
- Total P&L: ${trades_summary.get('profit_loss', 0):.2f}
- Average Win: ${trades_summary.get('avg_win', 0):.2f}
- Average Loss: ${trades_summary.get('avg_loss', 0):.2f}
- Best Win: ${trades_summary.get('best_win', 0):.2f}
- Worst Loss: ${trades_summary.get('worst_loss', 0):.2f}

Provide brief, actionable feedback in these areas:

1. **Performance Overview** (2 sentences): Overall assessment
2. **Strengths** (2 bullet points): What's working well
3. **Areas for Improvement** (2 bullet points): Specific suggestions
4. **Next Steps** (1-2 sentences): Actionable recommendations

Be encouraging but honest."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a trading coach providing constructive performance feedback."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ Unable to generate insights: {str(e)}"
