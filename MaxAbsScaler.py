import pandas as pd
from sklearn.preprocessing import MaxAbsScaler



data = pd.read_csv('iris.csv')
df = data.iloc[1:, :-1]

# MaxAbsScaler ölçekleyiciyi oluşturulması
scaler = MaxAbsScaler()

# Veri setini MaxAbsScaler ile ölçeklendirilmesi
scaled_data = scaler.fit_transform(df)

# Ölçeklenmiş veriyi DataFrame'e dönüştürülmesi
maxAbs_scaled_df = pd.DataFrame(scaled_data, columns=df.columns)


