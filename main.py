# Essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # loading dataset
    # url = 'https://drive.google.com/file/d/1O6PAzQd808rWNxkyL3ToO3qpaBlvQ2zC/view?usp=share_link'
    # path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
    # df = pd.read_csv(path, index_col=0)
    df = pd.read_csv("gpt-tweet-sentiment.csv", index_col=0)
    print(df)

    # analyzing dataset
    #Example EDA: Sentiment distribution
    sns.countplot(data=df, x="labels")
    plt.title('Distribution of Sentiment Classes')
    plt.show()

    # Your analysis...
    print("done")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
