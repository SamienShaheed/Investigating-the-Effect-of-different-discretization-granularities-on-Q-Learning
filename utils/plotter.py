import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def stability(rewards):
    # using relative error to measure the avg rate of change of rewards per 100 episodes,
    # to understand the stability of the algorithm
    # 1 - |(x2 - x1) - (x3 - x2)| / (x2 - x1))
    stability = 0
    for i in range(1, len(rewards) - 1):
        delta1 = abs(rewards[i] - rewards[i - 1]) / 100
        delta2 = abs(rewards[i + 1] - rewards[i]) / 100
        stability += abs(delta2 - delta1) / abs(delta2)

    # len(rewards) + 2 because we skipped the first and last rewards in the loop
    avg_stability = stability / (len(rewards) + 2)
    return avg_stability * 100

def output(rewards, times, total_time, epsilon, entropy, coverage, granularity='1x', run="1"):
    results = f"""Granularity: {granularity}
    Epsilon: {epsilon}
    Total time: {total_time:0.7f}s
    Best average reward: {rewards[-1]}
    Best time: {times[-1]}s
    Entropy: {entropy}
    Coverage: {coverage}
    Stability: {stability(rewards)}%"""

    print(results)

    here = os.path.dirname(os.path.realpath(__file__))
    root_directory = os.path.dirname(here)
    results_directory = os.path.join(root_directory, "results")
    run_directory = os.path.join(results_directory, f"run{run}")
    subdirectory = os.path.join(run_directory, granularity)
    filepath = os.path.join(run_directory, f"{granularity}_experiment_results.csv")

    if not os.path.isdir(run_directory):
        os.mkdir(run_directory)
    if not os.path.isdir(subdirectory):
        os.mkdir(subdirectory)

    with open(filepath, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Rewards', 'Times'])
        for reward, time in zip(rewards, times):
            csvwriter.writerow([reward, time])

    # Plot Rewards
    plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig(os.path.join(subdirectory, f"{granularity}_rewards.jpg"))
    plt.close()

    # Plot Time
    plt.plot(100 * (np.arange(len(rewards)) + 1), times, color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Average Time (in seconds)')
    plt.title('Average Time vs Episodes')
    plt.savefig(os.path.join(subdirectory, f"{granularity}_times.jpg"))
    plt.close()

    with open(os.path.join(here, f"all_experiment_results.txt"), 'a') as f:
        print(f"\n\nRun: {run}:\n\n", file=f)
        print(f"{results}", file=f)

    return results