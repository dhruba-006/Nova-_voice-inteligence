import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

# Configure Gemini
genai.configure(api_key=api_key)

print("\n" + "="*60)
print("FINDING MODELS THAT SUPPORT generateContent")
print("="*60 + "\n")

compatible_models = []

try:
    models = genai.list_models()
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            compatible_models.append(model.name)
            print(f"✓ {model.name}")
            print(f"  Display: {model.display_name}")
    
    print(f"\n\nTotal compatible models: {len(compatible_models)}\n")
    
    # Test first few compatible models
    print("=" * 60)
    print("TESTING COMPATIBLE MODELS")
    print("=" * 60 + "\n")
    
    test_query = "what is 2 plus 2"
    
    for model_name in compatible_models[:5]:  # Test first 5
        try:
            print(f"Testing: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(test_query)
            reply = response.text.strip() if hasattr(response, 'text') else "No response"
            print(f"✓ SUCCESS! Response: {reply[:100]}")
            print(f"*** USE THIS MODEL: {model_name} ***\n")
            break
        except Exception as e:
            print(f"✗ Failed: {str(e)[:80]}\n")
            
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
