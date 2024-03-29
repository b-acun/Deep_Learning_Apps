import pandas as pd
from sklearn.preprocessing import RobustScaler

data = pd.read_csv('iris.csv')
df = data.iloc[1:, :-1]

# RobustScaler ölçekleyiciyi oluşturulması
scaler = RobustScaler()

# Veri setini RobustScaler ile ölçeklendirilmesi
scaled_data = scaler.fit_transform(df)

# Ölçeklenmiş veriyi DataFrame'e dönüştürülmesi
robus_scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

# Ölçeklenmiş veriyi göster
print("RobustScaler ile Ölçeklenmiş Veri:")
print(robus_scaled_df)


