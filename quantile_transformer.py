import pandas as pd
from sklearn.preprocessing import QuantileTransformer

data = pd.read_csv('iris.csv')
df = data.iloc[1:, :-1]


# QuantileTransformer dönüştürücüsünü oluşturulması
transformer = QuantileTransformer()

# Veri setini QuantileTransformer ile ölçeklendirilmesi
transformed_data = transformer.fit_transform(df)

# Dönüştürülmüş veriyi DataFrame'e dönüştürülmesi
quantile_transformed_df = pd.DataFrame(transformed_data, columns=df.columns)

# Dönüştürülmüş veriyi göster
print("QuantileTransformer ile Dönüştürülmüş Veri:")
print(quantile_transformed_df)
