import seaborn as sns
import matplotlib.pyplot as plt


def visua_hist(data, title, xlabel, ylabel):
    # Vẽ histogram
    sns.histplot(data, bins=30, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def visua_boxplot(data, title, xlabel, ylabel):
    # Vẽ boxplot
    plt.boxplot(data, patch_artist=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)


def visua_scatter(data_x, data_y, title, xlabel, ylabel):
    # Vẽ scatter plot
    plt.scatter(data_x, data_y, alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)    
    plt.show()