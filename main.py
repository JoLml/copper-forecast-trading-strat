import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'copper-forecast-trading-strat')))


# Add project root and submodules to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "models"))
sys.path.append(os.path.join(project_root, "utils"))

from models.trainer import train_random_forest
from models.backtest import run_backtest

def main():
    print("🔧 Training model...")
    train_random_forest()
    
    print("\n🔁 Running backtest...")
    run_backtest()

if __name__ == "__main__":
    main()
