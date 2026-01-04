# /// script
# dependencies = ["wandb"]
# /// Only works because of uv. Wow.
import random
import wandb

wandb.init(project="wandb-test")
for _ in range(100):
    wandb.log({"test_metric": random.random()})