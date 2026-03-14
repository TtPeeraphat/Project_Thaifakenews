#!/usr/bin/env python
"""
Setup Script for TrueCheck AI
Installs all dependencies and verifies the setup is correct.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n📦 {description}...")
    print(f"   Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print(f"   ✅ Success")
            return True
        else:
            print(f"   ❌ Failed")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ⏱️  Timeout (took too long)")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Run setup process"""
    print("\n" + "=" * 60)
    print("🚀 TrueCheck AI - SETUP & VERIFICATION")
    print("=" * 60)
    
    steps = [
        ("Upgrade pip", "python -m pip install --upgrade pip -q"),
        ("Install requirements", "pip install -r requirements.txt -q"),
        ("Verify Python packages", "python -c \"import streamlit; import supabase; import torch; print('✅ All packages available')\""),
        ("Check environment file", "python -c \"from config import config; print('✅ .env loaded successfully')\""),
        ("Test database connection", "python -c \"from database_ops import get_supabase; get_supabase(); print('✅ Supabase connection works')\""),
    ]
    
    results = []
    
    print("\n📋 Starting setup steps...\n")
    
    for description, cmd in steps:
        success = run_command(cmd, description)
        results.append((description, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SETUP SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {description}")
    
    print(f"\n📈 Result: {passed}/{total} steps completed")
    
    if passed == total:
        print("\n✨ Setup complete! You're ready to use TrueCheck AI.")
        print("\n🚀 To start the app, run:")
        print("   streamlit run frontend.py")
        print("\n💡 To login, use:")
        print("   - Username: admin (if exists)")
        print("   - Or create a new account using Register button")
        
        print("\n🔧 If login fails, run diagnostic:")
        print("   python diagnose_db.py")
        
        return 0
    else:
        print(f"\n⚠️  {total - passed} step(s) failed. See errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the project directory")
        print("2. Verify .env file exists with correct values")
        print("3. Check internet connection to Supabase")
        print("4. Try running: pip install -r requirements.txt --force-reinstall")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
