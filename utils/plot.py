import matplotlib.pyplot as plt
import os


def plot_curves(history, save_dir=None, show=True):
    epochs = range(1, len(history["loss"]) + 1)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # =========================
    # 🎯 Loss（通用）
    # =========================
    plt.figure()
    plt.plot(epochs, history["loss"])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    if save_dir:
        plt.savefig(os.path.join(save_dir, "loss.png"))
    if show:
        plt.show()
    else:
        plt.close()

    # =========================
    # 🎯 F1（自适应）
    # =========================
    if "f1_micro" in history:  # multilabel
        plt.figure()
        plt.plot(epochs, history["f1_micro"])
        plt.plot(epochs, history["f1_macro"])
        plt.title("F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.legend(["F1-micro", "F1-macro"])

        if save_dir:
            plt.savefig(os.path.join(save_dir, "f1.png"))
        if show:
            plt.show()
        else:
            plt.close()

    elif "f1" in history:  # classification
        plt.figure()
        plt.plot(epochs, history["f1"])
        plt.title("F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.legend(["F1"])

        if save_dir:
            plt.savefig(os.path.join(save_dir, "f1.png"))
        if show:
            plt.show()
        else:
            plt.close()

    # =========================
    # 🎯 Accuracy（自适应）
    # =========================
    if "subset_acc" in history:  # multilabel
        plt.figure()
        plt.plot(epochs, history["subset_acc"])
        plt.plot(epochs, history["hamming_acc"])
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["Subset Acc", "Hamming Acc"])

    elif "acc" in history:  # classification
        plt.figure()
        plt.plot(epochs, history["acc"])
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["Accuracy"])

    else:
        return  # 没有可画的

    if save_dir:
        plt.savefig(os.path.join(save_dir, "accuracy.png"))
    if show:
        plt.show()
    else:
        plt.close()


def plot_missing_curve(results, save_dir=None, show=True):
    x = results["drop_prob"]
    y = results["f1_micro"]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel("Missing Rate")
    plt.ylabel("F1 Score")
    plt.title("Robustness under Modality Missing")

    if save_dir:
        plt.savefig(os.path.join(save_dir, "missing_curve.png"))

    if show:
        plt.show()
    else:
        plt.close()
