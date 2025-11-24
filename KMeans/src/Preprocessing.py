from google.colab import drive
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

drive.mount("/content/drive")
datasheet = "/content/drive/My Drive/Datasheet/DDoS/unlabeled_data.csv"
df = pd.read_csv(datasheet)
df.head(5)
df.describe()
df.drop(columns='Unnamed: 0', inplace=True)
df.head(2)
df.columns

Features = ['dt','dur', 'tot_dur','pktrate','port_no','rx_kbps','tot_kbps']
X = df[Features]

# Menghapus kolom negatif
kolom_negatif = ['dt','dur','tot_dur']
X = X[(X[kolom_negatif]> 0 ).all(axis=1)]
X = X.reset_index(drop=True)

# Menyamakan index df dengan index X
df = df.loc[X.index]
df = df.reset_index(drop=True)