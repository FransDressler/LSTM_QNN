#!/usr/bin/env python3
"""
🚀 Complete Stock Prediction Pipeline
====================================

Runs the complete stock prediction workflow:
1. Preprocessing (relative features)
2. Training (LSTM with company features)
3. Testing (on unseen data)

Author: Claude & Frans
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run command and handle errors"""
    print(f"\n🔄 {description}...")
    print(f"Running: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='src/stocks')

        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print(f"❌ {description} failed!")
            print("Error:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

    return True

def main():
    print("🚀 STOCK PREDICTION PIPELINE")
    print("=" * 50)
    print("This will run the complete stock prediction workflow:")
    print("1. ⚖️  Preprocessing: Create relative feature datasets")
    print("2. 🧠 Training: Train LSTM with company features")
    print("3. 📈 Testing: Test on unseen stock data")

    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Step 1: Preprocessing
    success = run_command("python preprocessing.py", "Preprocessing stock data")
    if not success:
        print("❌ Pipeline stopped at preprocessing")
        return

    # Step 2: Training
    success = run_command("python training.py", "Training LSTM model")
    if not success:
        print("❌ Pipeline stopped at training")
        return

    # Step 3: Testing
    success = run_command("python testing.py", "Testing model")
    if not success:
        print("❌ Pipeline stopped at testing")
        return

    print(f"\n🎉 PIPELINE COMPLETE!")
    print(f"✅ All steps completed successfully")
    print(f"📁 Check the following for results:")
    print(f"   📊 data/ - Training datasets")
    print(f"   🧠 models/ - Trained models and predictions")
    print(f"   📈 models/stock_predictions.png - Final results")

if __name__ == "__main__":
    main()