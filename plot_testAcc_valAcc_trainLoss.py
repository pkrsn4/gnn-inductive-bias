
#!/usr/bin/env python
# coding: utf-8

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [15, 20]
figs_loc = "figs/"


def plotAll (FIGURE_PREFIX, epoch, all_train_loss, all_train_acc, all_val_acc, all_test_acc, SAVE_FIG=False):
    x1 = np.array(range(1, epoch+1))
    train_loss = all_train_loss
    plt.plot(x1, train_loss, linestyle='-', label='Train Loss')

    title = "{} Train Loss".format(FIGURE_PREFIX)
    fig_name = "{}_train_loss.png".format(FIGURE_PREFIX)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.title(title)
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig(figs_loc+fig_name)
    plt.show()

    ##  ACCURACY

    plt.plot(x1, all_train_acc, linestyle='-', linewidth=2, label='Train')
    plt.plot(x1, all_val_acc, linestyle='-', linewidth=2, label='Val')
    plt.plot(x1, all_test_acc, linestyle='-', linewidth=2, label='Test')

    fig_name = "{}_acc.png".format(FIGURE_PREFIX)


    title = "{} Accuracy".format(FIGURE_PREFIX)

    plt.ylim(top=1.05)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")

    plt.title(title)
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig(figs_loc+fig_name)
    plt.show()
    plt.close()

