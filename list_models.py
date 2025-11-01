import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

print(f"API Key: {api_key[:10]}..." if api_key else "No API key found!")

# Configure Gemini
genai.configure(api_key=api_key)

print("\n" + "="*60)
print("LISTING AVAILABLE GEMINI MODELS")
print("="*60 + "\n")

try:
    models = genai.list_models()
    for model in models:
        print(f"Model: {model.name}")
        print(f"  Display Name: {model.display_name}")
        print(f"  Description: {model.description}")
        print(f"  Supported Methods: {model.supported_generation_methods}")
        print("-" * 60)
except Exception as e:
    print(f"Error listing models: {str(e)}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
