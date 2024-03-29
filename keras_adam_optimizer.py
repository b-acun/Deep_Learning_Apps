import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

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
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

# Modelin derlenmesi
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğit
model.fit(X_train, y_train_encoded, epochs=50, batch_size=32, validation_split=0.1)

# Modeli değerlendir
loss, accuracy = model.evaluate(X_test, y_test_encoded)
print(f'Test setindeki doğruluk: {accuracy}')
