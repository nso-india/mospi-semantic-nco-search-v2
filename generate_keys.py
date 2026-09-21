import secrets
import sys


def generate_secret_key():
    """Generate Flask SECRET_KEY"""
    return secrets.token_hex(32)


def generate_api_key():
    """Generate API authentication key"""
    return secrets.token_urlsafe(32)


def main():
    print("=" * 60)
    print("Security Key Generator")
    print("=" * 60)
    print()
    
    # Generate SECRET_KEY
    print("1. Flask SECRET_KEY (for session encryption):")
    print(f"   {generate_secret_key()}")
    print()
    
    # Generate API keys
    print("2. API Keys (for client authentication):")
    num_keys = input("   How many API keys to generate? (default: 3): ").strip()
    try:
        num_keys = int(num_keys) if num_keys else 3
    except ValueError:
        num_keys = 3
    
    api_keys = []
    for i in range(num_keys):
        key = generate_api_key()
        api_keys.append(key)
        print(f"   Key {i+1}: {key}")
    
    print()
    print("=" * 60)
    print("Add these to your .env file:")
    print("=" * 60)
    print()
    print(f"SECRET_KEY={generate_secret_key()}")
    print(f"API_KEYS={','.join(api_keys)}")
    print()
    print("Security Notes:")
    print("- Keep these keys secret and secure")
    print("- Never commit .env to version control")
    print("- Rotate keys periodically")
    print("- Use different keys for dev/staging/production")
    print()


if __name__ == "__main__":
    main()
