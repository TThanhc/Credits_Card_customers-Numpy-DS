import seaborn as sns
import matplotlib.pyplot as plt


def visua_hist(data, title):
    # Vẽ histogram
    sns.histplot(data, bins=30, kde=True)
    plt.title(title)
    plt.show()


# plt.figure(figsize=(12,6))
# sns.heatmap(missing_mask, cbar=False)
# plt.title("Heatmap Missing Values")
# plt.xlabel("Columns")
# plt.ylabel("Rows")
# plt.show()