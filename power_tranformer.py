import pandas as pd
from sklearn.preprocessing import PowerTransformer

data = pd.read_csv('iris.csv')
df = data.iloc[1:, :-1]


# PowerTransformer dönüştürücüsünü oluşturulması
transformer = PowerTransformer()

# Veri setini PowerTransformer ile ölçeklendirilmesi
transformed_data = transformer.fit_transform(df)

# Dönüştürülmüş veriyi DataFrame'e dönüştürülmesi
power_transformed_df = pd.DataFrame(transformed_data, columns=df.columns)

# Dönüştürülmüş veriyi göster
print("PowerTransformer ile Dönüştürülmüş Veri:")
print(power_transformed_df)


