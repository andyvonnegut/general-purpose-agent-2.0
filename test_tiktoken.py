#!/usr/bin/env python3
import tiktoken

# Test if the gpt-5 model is recognized
model_name = "gpt-5-2025-08-07"
print(f"Testing tiktoken with model: {model_name}")
print(f"Tiktoken version: {tiktoken.__version__}")

try:
    tokenizer = tiktoken.encoding_for_model(model_name)
    print(f"✓ Success! Tokenizer found: {tokenizer.name}")

    # Test encoding
    test_text = "Hello, world!"
    tokens = tokenizer.encode(test_text)
    print(f"✓ Test encoding successful: '{test_text}' -> {len(tokens)} tokens")

except Exception as e:
    print(f"✗ Error: {e}")
    print("\nTrying fallback approach...")

    # Try to get the encoding directly (GPT-5 likely uses o200k_base like GPT-4o)
    try:
        tokenizer = tiktoken.get_encoding("o200k_base")
        print(f"✓ Fallback successful! Using encoding: {tokenizer.name}")

        # Test encoding
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text)
        print(f"✓ Test encoding successful: '{test_text}' -> {len(tokens)} tokens")
    except Exception as e2:
        print(f"✗ Fallback also failed: {e2}")
