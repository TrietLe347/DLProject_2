from pickle import GLOBAL

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import models,layers, optimizers
import tensorflow as tf
from manual_dropout import ManualDropout


manual_dropout_rate = 0.1
learning_rate = 0.001
lamda_l2 = 1e-4

np.random.seed(0)

x = (np.linspace(0,10,200)).reshape(-1,1)


y = np.sin(x) + 0.1 * np.random.randn(200,1)




split = int(0.8 * len(x))
X_train, X_val = x[:split], x[split:]
y_train, y_val = y[:split], y[split:]


print(X_train)
print(X_val)
print(y_train)
print(y_val)

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(1,)),
    ManualDropout(manual_dropout_rate),
    layers.Dense(64, activation='relu'),
    ManualDropout(manual_dropout_rate),
    layers.Dense(1)
])


optimizer = optimizers.Adam(learning_rate)

epochs = 200
train_losses = []
val_losses = []
mse_losses = []

patience = 10

best_val_loss = float('inf')
best_weights = None
wait = 0



for epoch in range(epochs):

    with tf.GradientTape() as tape:
        y_pred = model(X_train, training = True)
        mse_loss = tf.reduce_mean(tf.square(y_pred - y_train) ** 2)

        l2_loss = tf.add_n([
            tf.reduce_sum(tf.square(w))
            for w in model.trainable_variables
            if 'kernel' in w.name
        ])

        total_loss = mse_loss + lamda_l2 * mse_loss

        train_loss = total_loss
    gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    val_pred = model(X_val, training = False)
    val_loss = tf.reduce_mean((y_val - val_pred) ** 2)


    train_losses.append(train_loss.numpy())
    mse_losses.append(mse_loss.numpy())
    val_losses.append(val_loss.numpy())


    #early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = model.get_weights()
        wait = 0

    else:
        wait += 1


    if epoch % 2 == 0:
        print(f"Epoch {epoch:03d} | "
              f"Train loss {train_loss.numpy():.5f} | "
              f"Val loss {val_loss.numpy():.5f}")

    if wait >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch}")
        model.set_weights(best_weights)
        break


plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Overfitting on Synthetic Data')
plt.legend()
plt.grid(alpha=0.3)
plt.show()