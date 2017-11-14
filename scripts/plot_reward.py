import matplotlib.pyplot as plt
import pandas as pd
import sys

if __name__ == '__main__':
    
    reward_csv = sys.argv[1]
    data = int(sys.argv[2])

    df = pd.DataFrame.from_csv(reward_csv)

    if data == 0:    
        plt.figure()
        plt.plot(df['Step'], df['Value'])
        plt.title('TSP 50, Average Tour Length (Training)')
        plt.xlabel('Step')
        plt.ylabel('Average Tour Length')
        plt.show()
    else:
        # average every 1000
        vals = []
        i = 1
        s = 0
        for index, row in df.iterrows():
            if i % 100 == 0:
                vals.append(s /100.)
                s = 0
            s += row['Value']
            i += 1
        plt.figure()
        plt.plot(vals)
        plt.plot(xrange(10), [5.95 for _ in range(10)])
        plt.title('TSP 50, Average Tour Length (Validation)')
        plt.xlabel('Step')
        plt.ylabel('Average Tour Length')
        plt.show()
