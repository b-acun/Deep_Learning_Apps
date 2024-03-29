import numpy as np
import pandas as pd

# Iris veri setini yükle
data = pd.read_csv('iris.csv')
df = data.iloc[1:, :-1]


# Her bir özellik için minimum ve maksimum değerleri bul
min_values = np.min(df, axis=0)
max_values = np.max(df, axis=0)

# Min-max normalleştirme
normalized_data = (df - min_values) / (max_values - min_values)

# Normalleştirilmiş veriyi göster
print("Normalleştirilmiş Veri:")
print(normalized_data)

