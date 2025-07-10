#!/usr/bin/env python3
"""
Launcher script for GARCH Trading Strategy
Run this from anywhere on your system
"""

import os
import sys

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add to Python path
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Change to project directory
os.chdir(script_dir)

print("🚀 Starting GARCH Trading Strategy for Arjay Siega")
print("=" * 50)
print(f"Project Directory: {script_dir}")
print(f"Account: 94435704 (MetaQuotes-Demo)")
print("=" * 50)

# Import and run the main trading engine
try:
    from src.main import main
    import asyncio
    
    # Run the trading engine
    asyncio.run(main())
    
except KeyboardInterrupt:
    print("\n🛑 Trading stopped by user")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")