from tensorflow import keras
import random

encoder = keras.models.Sequential([keras.layers.Dense(2, input_shape=[3])])

decoder = keras.models.Sequential([keras.layers.Dense(3, input_shape=[2])])

autoencoder = keras.models.Sequential([encoder, decoder])

autoencoder.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=0.1))

random_1 = random.randrange(1, 100)
random_2 = random.randrange(1, 100)
random_3 = random.randrange(1, 100)

x_train = [[random_1, random_2, random_3]]

history = autoencoder.fit(x_train, x_train, epochs=20)
codings = encoder.predict(x_train)