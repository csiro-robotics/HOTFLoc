import torch
import numpy as np
# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    epoch_max = 150
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(1, 1))], lr=1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_max+1, eta_min=1e-6)
    lrs = []
    for epoch in range(epoch_max+1):
        optimizer.step()
        scheduler.step()
        # print(scheduler.get_last_lr())
        lrs.append(scheduler.get_last_lr())

    lrs = np.array(lrs)
    fig = plt.figure(figsize=[10,5])
    ax = fig.add_subplot()
    ax.plot(lrs)
    plt.show()