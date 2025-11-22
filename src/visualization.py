import seaborn as sns
import matplotlib.pyplot as plt


def visua_hist(data, title, xlabel, ylabel):
    # Vẽ histogram
    sns.histplot(data, bins=30, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def visua_bar(data_x, data_y, title):
    plt.figure()
    plt.bar(data_x, data_y)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.show()

def visua_boxplot(data, title, xlabel, ylabel):
    # Vẽ boxplot
    plt.boxplot(data, patch_artist=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)


def visua_pie(count, uni_data, title):
    plt.figure()
    plt.pie(count, labels=uni_data, autopct='%1.1f%%')
    plt.title(title)
    plt.show()