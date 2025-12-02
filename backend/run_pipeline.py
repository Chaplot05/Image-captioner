import os
import subprocess
import time

def run_step(command, step_name):
    print(f"\n{'='*60}")
    print(f"🚀 Starting Step: {step_name}")
    print(f"{'='*60}")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ {step_name} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} failed with error: {e}")
        exit(1)

def main():
    print("🔄 Restoring Data and Starting Training Pipeline...")
    
    # Step 1: Download Dataset
    run_step("python download_dataset.py", "Download Dataset")
    
    # Step 2: Augment Data
    run_step("python augment_captions.py", "Augment Data (Fast Mode)")
    
    # Step 3: Train Model
    run_step("python train.py", "Train Model")

if __name__ == "__main__":
    main()
