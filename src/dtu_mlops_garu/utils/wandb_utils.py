import matplotlib.pyplot as plt
import torch
import wandb
from sklearn.metrics import RocCurveDisplay


def log_images(model, img: torch.Tensor):
    img = normalize(img)
    images = [wandb.Image(img[i].detach().cpu(), caption=f"Image {i}") for i in range(min(5, len(img)))]
    wandb.log({"images": images})

    # add a plot of histogram of the gradients
    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
    wandb.log({"gradients": wandb.Histogram(grads.cpu())})


def normalize(img: torch.Tensor):
    img += img.view(-1, 1, 28 * 28).min(axis=2, keepdim=True).values.unsqueeze(-1)
    img /= img.view(-1, 1, 28 * 28).max(axis=2, keepdim=True).values.unsqueeze(-1)
    return img * 255


def log_ROC(targets, preds):
    fig, ax = plt.subplots(figsize=(8, 6))

    for class_id in range(10):
        one_hot = torch.zeros_like(targets)
        one_hot[targets == class_id] = 1
        RocCurveDisplay.from_predictions(
            one_hot,
            preds[:, class_id],
            name=f"ROC curve for {class_id}",
            plot_chance_level=(class_id == 2),
            ax=ax,
        )

    ax.set_title("ROC Curves for All Classes")

    wandb.log({"roc_curves": wandb.Image(fig)})
    plt.close(fig)  # close the plot to avoid memory leaks and overlapping figures
