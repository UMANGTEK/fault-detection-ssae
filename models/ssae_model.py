from keras import layers, models, regularizers

def build_autoencoder(input_dim, encoded_dim, sparsity=1e-5):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoded_dim, activation='relu',
                           activity_regularizer=regularizers.l1(sparsity))(input_layer)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = models.Model(input_layer, decoded)
    encoder = models.Model(input_layer, encoded)
    return autoencoder, encoder

