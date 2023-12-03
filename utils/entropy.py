import numpy as np

def calculate_entropy(action_distribution):
    probabilities = np.array(list(action_distribution.values())) / sum(action_distribution.values())
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy