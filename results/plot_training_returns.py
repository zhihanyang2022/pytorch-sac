import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage.filters import uniform_filter1d

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str)
args = parser.parse_args()

base_dir = f'results/training_returns_json/{args.env}'

num_seeds = len(os.listdir(base_dir))

returns = np.zeros((num_seeds, 1000))

for seed in range(1, num_seeds+1):

    with open(os.path.join(base_dir, f'{seed}.json')) as json_f:
        returns[seed-1] = uniform_filter1d(np.array(json.load(json_f)), size=20, mode='reflect')

plt.plot(np.arange(1000), np.mean(returns, axis=0))
plt.fill_between(np.arange(1000), np.max(returns, axis=0), np.min(returns, axis=0), alpha=0.3)
plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
plt.yticks([-500, -400, -300, -200, -100, 0])
plt.ylim(-500, 0)
plt.grid()

plt.xlabel('Episode')
plt.ylabel('Return')
plt.title(args.env)

plt.savefig(f'results/training_returns_svg/{args.env}.svg')
