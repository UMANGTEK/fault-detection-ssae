import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import os

# === Load merged dataset ===
df = pd.read_csv("all_circuits.csv")
X = np.array([[int(bit) for bit in row] for row in df["input_vector"]])
X = X.astype(np.float32)

# === Train/test split ===
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

# === Build sparse autoencoder ===
def build_autoencoder(input_dim, encoded_dim, sparsity=1e-5):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoded_dim, activation='relu',
                           activity_regularizer=regularizers.l1(sparsity))(input_layer)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = models.Model(inputs=input_layer, outputs=decoded)
    encoder = models.Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder, encoder

# AE1: 36 → 24
ae1, enc1 = build_autoencoder(X.shape[1], 24)
history1 = ae1.fit(X_train, X_train, epochs=50, batch_size=8, validation_data=(X_val, X_val))
X_enc1 = enc1.predict(X)

# AE2: 24 → 12
ae2, enc2 = build_autoencoder(24, 12)
history2 = ae2.fit(X_enc1, X_enc1, epochs=50, batch_size=8, validation_split=0.2)
X_enc2 = enc2.predict(X_enc1)

# AE3: 12 → 6
ae3, enc3 = build_autoencoder(12, 6)
history3 = ae3.fit(X_enc2, X_enc2, epochs=50, batch_size=8, validation_split=0.2)

# Save models
os.makedirs("models", exist_ok=True)
enc1.save("models/encoder1.h5")
enc2.save("models/encoder2.h5")
enc3.save("models/encoder3.h5")

# Plot loss curves
os.makedirs("report", exist_ok=True)
def plot_loss(history, title, filename):
    plt.figure()
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history.get('val_loss', []), label='Val')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"report/{filename}")
    plt.close()

plot_loss(history1, "AE1 Loss", "loss_ae1.png")
plot_loss(history2, "AE2 Loss", "loss_ae2.png")
plot_loss(history3, "AE3 Loss", "loss_ae3.png")

print("✅ SSAE trained and saved.")
