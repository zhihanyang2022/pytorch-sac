import json

import wandb
api = wandb.Api()
run = api.run("yangz2/offline-rl/3h1q8d6q")

with open('../results/training_returns_json/Pendulum-v0/3.json', 'w+') as json_f:
    json.dump(list(run.history()['return']), json_f)