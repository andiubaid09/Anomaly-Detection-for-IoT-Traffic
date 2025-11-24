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