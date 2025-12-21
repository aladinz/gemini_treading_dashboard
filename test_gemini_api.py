"""
Simple script to test Google Gemini API key
"""
import os

# Set your API key here or use environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your_gemini_api_key_here')  # Replace with your actual API key

try:
    import google.generativeai as genai
    
    # Configure the API
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Create a simple model instance
    model = genai.GenerativeModel('gemini-pro')
    
    # Test with a simple prompt
    print("Testing Gemini API connection...")
    response = model.generate_content("Say 'Hello! API is working!' in one short sentence.")
    
    print("\n✅ SUCCESS! Gemini API is working!")
    print(f"\nResponse: {response.text}")
    print(f"\nYour API key is valid and active.")
    
except ImportError:
    print("❌ ERROR: google-generativeai library not installed")
    print("\nTo install, run:")
    print("pip install google-generativeai")
    
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
    print("\nPossible issues:")
    print("1. Invalid API key")
    print("2. API key expired")
    print("3. Network connection issues")
    print("4. API quota exceeded")
