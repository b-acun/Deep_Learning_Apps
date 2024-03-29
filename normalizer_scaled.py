import pandas as pd
from sklearn.preprocessing import Normalizer

data = pd.read_csv('iris.csv')
df = data.iloc[1:, :-1]

# Normalizer ölçekleyiciyi oluşturulması
scaler = Normalizer()

# Veri setini Normalizer ile ölçeklendirilmesi
scaled_data = scaler.fit_transform(df)

# Ölçeklenmiş veriyi DataFrame'e dönüştürülmesi
normalizer_scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

# Ölçeklenmiş veriyi göster
print("Normalizer ile Ölçeklenmiş Veri:")
print(normalizer_scaled_df)
