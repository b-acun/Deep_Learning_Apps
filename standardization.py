import numpy as np
import pandas as pd

# Iris veri setini yükle
data = pd.read_csv('iris.csv')
df = data.iloc[1:, :-1]

# Her bir özellik için ortalamayı ve standart sapmayı hesapla
mean_values = np.mean(df, axis=0)
std_values = np.std(df, axis=0)

# Standardizasyon
standardized_data = (df - mean_values) / std_values

# Normalleştirilmiş veriyi göster
print("Standartlaştırılmış Veri:")
print(standardized_data)
