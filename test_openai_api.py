"""
Simple script to test OpenAI API key
"""
import os

# Set your API key here or use environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your_openai_api_key_here')  # Replace with your actual API key

try:
    from openai import OpenAI
    
    # Initialize the client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Test with a simple prompt
    print("Testing OpenAI API connection...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say 'Hello! API is working!' in one short sentence."}
        ],
        max_tokens=50
    )
    
    print("\n✅ SUCCESS! OpenAI API is working!")
    print(f"\nResponse: {response.choices[0].message.content}")
    print(f"\nYour API key is valid and active.")
    print(f"Model used: {response.model}")
    
except ImportError:
    print("❌ ERROR: openai library not installed")
    print("\nTo install, run:")
    print("pip install openai")
    
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
    print("\nPossible issues:")
    print("1. Invalid API key")
    print("2. API key expired")
    print("3. Network connection issues")
    print("4. API quota exceeded")
    print("5. Insufficient credits/billing not set up")
