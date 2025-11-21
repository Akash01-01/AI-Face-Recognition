#!/usr/bin/env python
"""
Production Deployment Health Check Script
Monitor the face recognition system status on Render.com
"""

import requests
import time
import json
from datetime import datetime

PRODUCTION_URL = "https://ai-face-recognition-cw0d.onrender.com"
CHECK_INTERVAL = 10  # seconds between checks

def check_health():
    """Check the /health endpoint"""
    try:
        response = requests.get(f"{PRODUCTION_URL}/health", timeout=30)
        if response.status_code == 200:
            data = response.json()
            return True, data
        else:
            return False, {"error": f"Status {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

def check_debug():
    """Check the /debug endpoint for detailed info"""
    try:
        response = requests.get(f"{PRODUCTION_URL}/debug", timeout=30)
        if response.status_code == 200:
            data = response.json()
            return True, data
        else:
            return False, {"error": f"Status {response.status_code}", "text": response.text[:200]}
    except Exception as e:
        return False, {"error": str(e)}

def main():
    """Monitor production deployment"""
    print("=" * 60)
    print("Production Deployment Monitor")
    print(f"URL: {PRODUCTION_URL}")
    print("=" * 60)
    print()
    
    attempt = 1
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Check #{attempt}")
        print("-" * 60)
        
        # Check health endpoint
        health_ok, health_data = check_health()
        print(f"Health Endpoint: {'✅ OK' if health_ok else '❌ FAILED'}")
        if health_ok:
            print(f"  Status: {health_data.get('status', 'Unknown')}")
            print(f"  OpenCV: {health_data.get('opencv', 'Unknown')}")
        else:
            print(f"  Error: {health_data.get('error', 'Unknown')}")
        
        # Check debug endpoint
        debug_ok, debug_data = check_debug()
        print(f"\nDebug Endpoint: {'✅ OK' if debug_ok else '❌ FAILED'}")
        if debug_ok:
            print(f"  Status: {debug_data.get('status', 'Unknown')}")
            print(f"  OpenCV Available: {debug_data.get('opencv_available', 'Unknown')}")
            print(f"  Face Module Available: {debug_data.get('lbph_module_available', 'Unknown')}")
            print(f"  Face Recognizer Loaded: {debug_data.get('face_recognizer_loaded', 'Unknown')}")
            print(f"  Employees in DB: {debug_data.get('employees_in_db', 'Unknown')}")
            print(f"  Images in DB: {debug_data.get('images_in_db', 'Unknown')}")
            print(f"  Trained Labels: {debug_data.get('trained_labels', 'Unknown')}")
            
            # Check if system is ready
            if debug_data.get('face_recognizer_loaded') and debug_data.get('trained_labels', 0) > 0:
                print("\n🎉 SYSTEM IS READY AND OPERATIONAL!")
                break
            else:
                print("\n⏳ System is still initializing...")
        else:
            print(f"  Error: {debug_data.get('error', 'Unknown')}")
        
        print(f"\nWaiting {CHECK_INTERVAL} seconds before next check...")
        attempt += 1
        time.sleep(CHECK_INTERVAL)
        
        # Stop after 30 attempts (5 minutes)
        if attempt > 30:
            print("\n⚠️ Stopped after 30 attempts. System may need manual inspection.")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
