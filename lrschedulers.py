import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import _LRScheduler

class NoamLR(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1, max_lr=0.001):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        lr = self.d_model ** -0.5 * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)
        return [min(base_lr * lr, self.max_lr) for base_lr in self.base_lrs]

if __name__ == "__main__":
    # Example Usage:
    optimizer = torch.optim.AdamW(params=torch.nn.ParameterList([torch.nn.Parameter(torch.randn(10, 10))]), lr=1)  # set a large initial lr as it'll be adjusted by the scheduler
    scheduler = NoamLR(optimizer, d_model=128, warmup_steps=4000)

    lrs = []
    for epoch in range(10000):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # Plotting the learning rate schedule
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Noam Learning Rate Scheduler')
    plt.show()
