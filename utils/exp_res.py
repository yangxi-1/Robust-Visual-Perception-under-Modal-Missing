from datetime import datetime
import os


def create_exp_dir(base="outputs", dataset="mmdb"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(base, f"exp_{dataset}_dropSweep_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir
