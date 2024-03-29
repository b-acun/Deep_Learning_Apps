import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

# Verisetinin yüklenmesi
df = pd.read_csv('iris.csv')

X_data = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] # giriş verileri
Y_data = df['Species'] #çıkış verileri(etiketler)

#Veri setinin eğitim ve test verisi olarak ayrılması
X_train, X_test, y_train, y_test = train_test_split(X_data.values, Y_data, test_size = 0.2, random_state=42)

# Etiketlerin one-hot encoding'e dönüştürülmesi (çıktıların 0,1,2 olarak ayrılması)
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Modeli oluşturulması
model = Sequential([
    Dense(12, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

#Veri setinin eğitim ve test verisi olarak ayrılması
X_train, X_test, y_train, y_test = train_test_split(X_data.values, Y_data, test_size = 0.2, random_state=42)

# Etiketlerin one-hot encoding'e dönüştürülmesi (çıktıların 0,1,2 olarak ayrılması)
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Modeli oluşturulması
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])