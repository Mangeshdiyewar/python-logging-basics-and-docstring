import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
import numpy as np
import logging
def prepare_data(df, target_col="y"):
    """
    The prepare_data function prepares the data for training.
    It drops the target column and returns x, y where x is a pandas DataFrame of features and y is a pandas Series of target values

    :param df (pd.DataFrame: this is the data frame
    :param target_col (str,optional) : label col name default to 'y'
    :return: tuple: label and x
    :doc-author: Mangesh
    """

    logging.info(("preparing the data for training"))
    x = df.drop(target_col, axis=1)

    y = df['y']

    return x, y


# defining the function to show why perceptron model is work only for liner dataset

def save_plot(df, model, filename="plot.png", plot_dir="plots"):
    def _create_base_plot(df):
        logging.info("creating base plot")
        df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="coolwarm")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)

        figure = plt.gcf()
        figure.set_size_inches(10, 8)

    def _plot_decision_regions(x, y, classifier, resolution=0.02):
        logging.info("plotting the decision regions")
        colors = ("cyan", "lightgreen")
        cmap = ListedColormap(colors)

        x = x.values  # as an array
        x1 = x[:, 0]
        x2 = x[:, 1]

        x1_min, x1_max = x1.min() - 1, x1.max() + 1
        x2_min, x2_max = x2.min() - 1, x2.max() + 1

        # for mesh grid
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x1_max, resolution)
                               )

        y_hat = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        y_hat = y_hat.reshape(xx1.shape)

        plt.contourf(xx1, xx2, y_hat, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        plt.plot

    x, y = prepare_data(df)

    _create_base_plot(df)
    _plot_decision_regions(x, y, model)

    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, filename)
    plt.savefig(plot_path)
    logging.info(f"saving the plot at {plot_path}")