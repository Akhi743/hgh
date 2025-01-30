from Constants import Constants
from Experiments import Experiments

if __name__ == '__main__':
    print("Running DR-VIDAL for Social Risk Score Prediction")
    running_mode = "original_data"
    experiment = Experiments(running_mode)
    experiment.run_all_experiments(iterations=5)  # Can adjust number of iterations as needed