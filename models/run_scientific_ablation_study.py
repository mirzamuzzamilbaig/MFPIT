import subprocess
import sys
import os

def run_cmd(cmd):
    print(f"\n[ORCHESTRATOR] Running: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Print output in real time to logging
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
        sys.stdout.flush()
        
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        print(f"[ORCHESTRATOR] Error: Command '{cmd}' failed with return code {return_code}")
        sys.exit(return_code)

def main():
    # Enforce working directory is correct
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=================================================================")
    print("--- MFPIT TRUE ABLATION STUDY SEQUENTIAL RUNNER ---")
    print("=================================================================")
    
    # 1. Train terrain_only
    run_cmd("python run_ablations.py --ablation terrain_only --epochs 25")
    
    # 2. Train no_jrc
    run_cmd("python run_ablations.py --ablation no_jrc --epochs 25")
    
    # 3. Train dynamic_only
    run_cmd("python run_ablations.py --ablation dynamic_only --epochs 25")
    
    # 4. Evaluate all retrained ablations
    run_cmd("python evaluate_ablations.py")
    
    print("\n[ORCHESTRATOR] True Retraining Ablation study completed successfully!")
    print("=================================================================")

if __name__ == "__main__":
    main()
