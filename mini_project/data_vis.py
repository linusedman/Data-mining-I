import pandas as pd
from seaborn import pairplot
import matplotlib.pyplot as plt



data = pd.read_csv("data.csv")

pairplot(data, hue= "genre")
plt.show()
# plt.savefig("spotify_api_adverture/pairplot_song_data.png")


