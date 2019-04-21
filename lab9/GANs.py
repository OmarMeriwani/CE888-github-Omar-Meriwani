import keras
from keras import layers
import numpy as np

latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = keras.Input(shape=(latent_dim,))

# Transforms the input into a 16 × 16 128-channel feature map
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# Upsamples to 32 × 32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# Produces a 32 × 32 1-channel feature map (shape of a CIFAR10 image)
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)

# Instantiates the generator model, which maps the input of shape (latent_dim,)
#into an image of shape (32, 32, 3)
generator = keras.models.Model(generator_input, x)
generator.summary()

discriminator_input = layers.Input(shape=(height, width, channels))

x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

# One dropout layer: an important trick
x = layers.Dropout(0.4)(x)

# Classification layer
x = layers.Dense(1, activation='sigmoid')(x)

#Instantiates the discriminator model, which turns a (32, 32, 3) input into a binary
#classification decision (fake/real)
discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

# Uses gradient clipping (by value) in the optimizer and To stabilize training,
# uses learning-rate decay
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008,clipvalue=1.0,decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy')
# Sets discriminator weights to non-trainable (this will only apply to the gan model)
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

import os
from keras.preprocessing import image

# Loads CIFAR10 data
(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()

# Selects frog images (class 6)
x_train = x_train[y_train.flatten() == 6]

x_train = x_train.reshape(
    (x_train.shape[0],) +
    (height, width, channels)).astype('float32') / 255.

iterations = 10000
batch_size = 20
save_dir = ''  # Specifies where you want to save generated images

start = 0
# Samples random points in the latent space
for step in range(iterations):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

# Decodes them to fake images
generated_images = generator.predict(random_latent_vectors)

# Combines them with real images
stop = start + batch_size
real_images = x_train[start: stop]
combined_images = np.concatenate([generated_images, real_images])

labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
labels += 0.05 * np.random.random(labels.shape)

# Trains the discriminator
d_loss = discriminator.train_on_batch(combined_images, labels)

# Samples random points in the latent space
random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

# Assembles latent space labels that say “these are all real images” (it’s a lie!)
misleading_targets = np.zeros((batch_size, 1))

# Trains the generator (via the gan model, where the discriminator weights are frozen)
a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

start += batch_size
if start > len(x_train) - batch_size:
    start = 0

if step % 100 == 0:
    gan.save_weights('gan.h5')

print('discriminator loss:', d_loss)
print('adversarial loss:', a_loss)

img = image.array_to_img(generated_images[0] * 255., scale=False)
img.save(os.path.join(save_dir,'generated_frog' + str(step) + '.png'))

img = image.array_to_img(real_images[0] * 255., scale=False)
img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))