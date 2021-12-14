import seaborn as sns

import matplotlib.pyplot as plt

if __name__ == '__main__':

    ax = sns.kdeplot([100000, 1, 1, 0], color="Red", shade=True)
    plt.show()
