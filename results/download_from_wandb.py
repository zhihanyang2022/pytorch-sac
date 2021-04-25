import json

import wandb
api = wandb.Api()
run = api.run("yangz2/offline-rl/2fn8w3tz")

history = run.scan_history()

with open('results/training_returns_json/Pendulum-v0/4.json', 'w+') as json_f:
    json.dump([row["return"] for row in history], json_f)