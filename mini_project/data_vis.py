import pandas as pd
from seaborn import pairplot
import matplotlib.pyplot as plt

data = pd.read_csv('Data-mining-I/mini_project/raw_data.csv')

# colums to drop: key, mode, time_signature, duration_ms, 
df = data.drop(columns=['key', 'mode', 'time_signature', 'duration_ms'])

# df.to_csv('Data-mining-I/mini_project/data.csv', index=False)
# pairplot(df, hue= "genre")
# plt.savefig('Data-mining-I/mini_project/song_data.png')
# plt.show()