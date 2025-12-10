import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('./eval/summary_statistics.csv', index_col=False)


    names = df['demo_name']
    print(names)
    x = np.arange(len(names)/3)
    data_10 = df[df['priming_value'] == 10]
    data_20 = df[df['priming_value'] == 20]
    data_30 = df[df['priming_value'] == 30]
    success_rates_10 = data_10['success_rate'].values
    success_rates_20 = data_20['success_rate'].values
    success_rates_30 = data_30['success_rate'].values
    w = 0.3

    plt.bar(x-w, success_rates_10, width=w, label='priming value = 10', color='red', edgecolor='white')
    plt.bar(x, success_rates_20, width=w, label='priming value = 20', color='orange', edgecolor='white')
    plt.bar(x + w, success_rates_30, width=w, label='priming value = 30', color='blue', edgecolor='white')
    plt.xticks(x, ['20G', '20B'])
    plt.ylim([0, 100])
    plt.xlabel('Demo')
    plt.ylabel('Average Success Rate')
    plt.legend(loc='upper center')
    plt.show()
    plt.savefig('SuccessRate.png')

if __name__ == "__main__":
    main()