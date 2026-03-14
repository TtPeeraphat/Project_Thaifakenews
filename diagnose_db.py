#!/usr/bin/env python
"""
Database Connection Diagnostic Tool
Helps troubleshoot login and database connection issues.
"""

import os
import sys
from pathlib import Path

def check_env_file():
    """Check if .env file exists and has required variables"""
    print("=" * 60)
    print("🔍 CHECKING ENVIRONMENT FILE (.env)")
    print("=" * 60)
    
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found!")
        print("   Create it: cp .env.example .env")
        return False
    
    print("✅ .env file exists")
    
    # Read .env
    env_vars = {}
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key] = value[:20] + "..." if len(value) > 20 else value
    
    # Check required variables
    required = ["SUPABASE_URL", "SUPABASE_KEY", "SENDER_EMAIL", "SENDER_PASSWORD"]
    missing = []
    
    for var in required:
        if var in env_vars:
            print(f"✅ {var} = {env_vars[var]}")
        else:
            print(f"❌ {var} = MISSING")
            missing.append(var)
    
    if missing:
        print(f"\n❌ Missing {len(missing)} required variables: {', '.join(missing)}")
        return False
    
    return True


def check_config_loading():
    """Check if config.py can load .env variables"""
    print("\n" + "=" * 60)
    print("🔍 CHECKING CONFIG LOADING")
    print("=" * 60)
    
    try:
        from config import config
        print("✅ config.py loaded successfully")
        
        # Check database config
        if config.database.supabase_url:
            print(f"✅ SUPABASE_URL loaded: {config.database.supabase_url[:30]}...")
        else:
            print("❌ SUPABASE_URL is empty")
            return False
        
        if config.database.supabase_key:
            print(f"✅ SUPABASE_KEY loaded: {config.database.supabase_key[:20]}...")
        else:
            print("❌ SUPABASE_KEY is empty")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False


def check_supabase_connection():
    """Test if we can connect to Supabase"""
    print("\n" + "=" * 60)
    print("🔍 CHECKING SUPABASE CONNECTION")
    print("=" * 60)
    
    try:
        from config import config
        from supabase import create_client
        
        url = config.database.supabase_url
        key = config.database.supabase_key
        
        print(f"🔗 Attempting to connect to: {url}")
        print(f"   Using key: {key[:20]}...")
        
        # Try to create client
        client = create_client(url, key)
        print("✅ Supabase client created successfully")
        
        # Try a simple query to test connectivity
        print("🔍 Testing database query...")
        result = client.table("users").select("id").limit(1).execute()
        print("✅ Database query successful!")
        print(f"   Response: {len(result.data)} rows returned")
        
        return True
        
    except OSError as e:
        if "11001" in str(e) or "getaddrinfo" in str(e):
            print(f"❌ [NETWORK ERROR] Cannot reach Supabase: {e}")
            print("   Troubleshooting steps:")
            print("   1. Check your internet connection")
            print("   2. Verify SUPABASE_URL in .env is correct (should be https://...)")
            print("   3. Check if Supabase service is accessible")
            print("   4. Try: curl -I https://orxtfxdernqmpkfmsijj.supabase.co")
        else:
            print(f"❌ [OS ERROR] {e}")
        return False
        
    except Exception as e:
        error_type = type(e).__name__
        print(f"❌ [{error_type}] Connection failed: {e}")
        
        if "invalid" in str(e).lower():
            print("   → Check SUPABASE_URL and SUPABASE_KEY in .env")
        elif "unauthorized" in str(e).lower():
            print("   → Check SUPABASE_KEY - it may be incorrect or expired")
        
        return False


def check_database_tables():
    """Check if required database tables exist"""
    print("\n" + "=" * 60)
    print("🔍 CHECKING DATABASE TABLES")
    print("=" * 60)
    
    try:
        from config import config
        from supabase import create_client
        
        client = create_client(config.database.supabase_url, config.database.supabase_key)
        
        # Check for required tables
        tables_to_check = ["users", "predictions", "feedbacks"]
        
        for table in tables_to_check:
            try:
                result = client.table(table).select("id").limit(1).execute()
                print(f"✅ Table '{table}' exists")
            except Exception as e:
                if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                    print(f"❌ Table '{table}' does not exist")
                else:
                    print(f"⚠️  Cannot verify table '{table}': {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cannot check tables: {e}")
        return False


def test_login():
    """Test login functionality"""
    print("\n" + "=" * 60)
    print("🔍 TESTING LOGIN")
    print("=" * 60)
    
    try:
        from database_ops import authenticate_user
        
        # Try with test credentials (will fail but shows if connection works)
        print("🔐 Attempting test authentication...")
        print("   Username: testuser")
        print("   Password: testpass")
        
        result = authenticate_user("testuser", "testpass")
        
        if result is None:
            print("✅ Authentication function works (returned None for invalid creds)")
            print("   This is expected - invalid credentials should return None")
            return True
        else:
            print(f"✅ Unexpected result: {result}")
            return True
        
    except OSError as e:
        if "11001" in str(e):
            print(f"❌ [NETWORK ERROR] Cannot reach Supabase during login: {e}")
        else:
            print(f"❌ [OS ERROR] {e}")
        return False
        
    except Exception as e:
        print(f"❌ Login test failed: {e}")
        return False


def main():
    """Run all diagnostic checks"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  DATABASE CONNECTION DIAGNOSTIC TOOL".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    checks = [
        ("Environment File", check_env_file),
        ("Config Loading", check_config_loading),
        ("Supabase Connection", check_supabase_connection),
        ("Database Tables", check_database_tables),
        ("Login Test", test_login),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ Unexpected error in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print(f"\n📈 Result: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✨ All checks passed! Your setup looks good.")
        print("   Try logging in again with correct credentials.")
    elif passed >= total - 1:
        print("\n⚠️  Most checks passed, but there may be minor issues.")
        print("   Review the errors above and try again.")
    else:
        print("\n❌ Multiple issues detected. See errors above for details.")
        print("   Common fixes:")
        print("   1. Verify .env file exists with correct values")
        print("   2. Check internet connection to Supabase")
        print("   3. Regenerate SUPABASE_KEY from Supabase dashboard")
    
    print("\n" + "=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
