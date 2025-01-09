# %%
import os

import numpy as np
import tensorflow as tf

# %%
data_path = "tf_data/beat_data"
INPUT_SIZE = 360
NUM_CLASSES = 17
# load npz
arrays = np.load(os.path.join(data_path, "data.npz"))
data = arrays["data"]
label = arrays["label"]
# set label to 0(normal) and 1(abnormal)
new_label = []
normal_count = 0
abnormal_count = 0
for i in range(len(label)):
    if label[i] == 9:
        new_label.append(0)
        normal_count += 1
    else:
        new_label.append(1)
        abnormal_count += 1
label = np.array(new_label)
print(f"Normal count: {normal_count}")
print(f"Abnormal count: {abnormal_count}")
NUM_CLASSES = 2
# %%
# convert label to one-hot
label = tf.keras.utils.to_categorical(label, num_classes=NUM_CLASSES)
# get array from npz
dataset = tf.data.Dataset.from_tensor_slices((data, label))
# with open(os.path.join(data_path, "label_list.txt"), "r") as f:
#     label_list = f.read()
# label_list = eval(label_list)
label_list = ["Normal", "Abnormal"]
dataset_size = dataset.reduce(0, lambda x, _: x + 1)

# Print the size of the dataset
print(f"The x shape of the dataset is {dataset.element_spec[0].shape}.")
print(f"The y shape of the dataset is {dataset.element_spec[1].shape}.")
print(f"The size of the dataset is {dataset_size}.")
# print(f"The labels in the dataset are {label_list}.")
# print(f"The number of labels is {len(label_list)}.")
# %%

# %%
split_ratio = 0.8
batch_size = 128


# shuffle the data
data_train, data_test = tf.keras.utils.split_dataset(
    dataset, left_size=split_ratio, shuffle=True
)
data_train = (
    data_train.cache()
    .shuffle(buffer_size=int(dataset_size), reshuffle_each_iteration=True)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
data_test = data_test.cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
dataset = dataset.cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
# %%
# Define the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(INPUT_SIZE,)),
        tf.keras.layers.Dense(units=2048, activation=tf.keras.layers.PReLU()),
        tf.keras.layers.Dense(units=1024, activation=tf.keras.layers.PReLU()),
        tf.keras.layers.Dense(units=512, activation=tf.keras.layers.PReLU()),
        tf.keras.layers.Dense(units=256, activation=tf.keras.layers.PReLU()),
        tf.keras.layers.Dense(units=64, activation=tf.keras.layers.PReLU()),
        tf.keras.layers.Dense(units=NUM_CLASSES, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

# Print the model summary
model.summary()
# %%
# Define the callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=10, verbose=1
    ),
]


# %%
# Train the model
model.fit(data_train, epochs=100, validation_data=data_test, callbacks=callbacks)

# %%
# Save the model
model.save("model.h5")

# %%
# draw confusion matrix with percentage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# get confusion matrix
plt_data = data_test

y_pred = model.predict(plt_data)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.concatenate([y for x, y in plt_data], axis=0)
y_true = np.argmax(y_true, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 10))

# draw confusion matrix with percentage
# cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
# sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", square=True)

# draw confusion matrix with count
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True)

plt.ylabel("Actual label")
plt.xlabel("Predicted label")
# plt.xticks(np.arange(NUM_CLASSES) + 0.5, label_list, rotation=90)
# plt.yticks(np.arange(NUM_CLASSES) + 0.5, label_list, rotation=0)
plt.title("Confusion matrix")
plt.show()
# %%
# prune the model
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 2
validation_split = 0.1  # 10% of training set will be used for validation set.
num_images = data.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
# Define model for pruning.
pruning_params = {
    "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50, final_sparsity=0.80, begin_step=0, end_step=end_step
    )
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)
# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)
model_for_pruning.summary()

# %%
# Fine-tune model with pruning
logdir = "logs"
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]
model_for_pruning.fit(
    data_train, epochs=10, validation_data=data_test, callbacks=callbacks
)
# %%
# Save the model
model_for_pruning.save("pruned_model.h5")
# %%
# draw confusion matrix with pruned model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# get confusion matrix
plt_data = data_test

y_pred = model_for_pruning.predict(plt_data)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.concatenate([y for x, y in plt_data], axis=0)
y_true = np.argmax(y_true, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 10))

# draw confusion matrix with percentage
# cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
# sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", square=True)

# draw confusion matrix with count
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True)

plt.ylabel("Actual label")
plt.xlabel("Predicted label")
# plt.xticks(np.arange(NUM_CLASSES) + 0.5, label_list, rotation=90)
# plt.yticks(np.arange(NUM_CLASSES) + 0.5, label_list, rotation=0)
plt.title("Confusion matrix")
plt.show()

# %%
# Convert the model to C++
from everywhereml.code_generators.tensorflow import convert_model

# %%
cpp_code = convert_model(model, data, label, model_name="ecgmodel")
# %%
with open("../esp32/ecgmodel.h", "w") as f:
    f.write(cpp_code)
# %%
