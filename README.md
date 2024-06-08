# AI4EO 
Video link：https://youtu.be/jjKA8vhYZA4?si=WiTXaYVvd1TiSKok

The code shown in the video is not the final version and differs slightly from the one in the GitHub repository, but the basic principles and sequence are consistent. This is because the final code has undergone several updates and optimizations.

## A description of the problem to be tacked

The aim of this project is to leverage artificial intelligence and IRIS to classify cloud and ocean data obtained from the Sentinel-3 satellite. By comparing the differences in classification images produced by various algorithms, this study evaluates the value and limitations of artificial intelligence in satellite remote sensing and Earth observation.

## A figure illustrating the remote sensing technique

![A figure illustrating the optical sensor and the 21 bands_page-0001 (4)](https://github.com/ZCBEZHU/AI4EOFINALERNESTO/assets/160029229/b8e4dbc9-6475-4a7c-8df5-9c9fe8f0b130)

## A diagram of the AI algorithm and its implementation


![Doc1 (1)](https://github.com/ZCBEZHU/AI4EOFINALERNESTO/assets/160029229/66708df7-1d92-42d5-9d36-008cabb70090)




## Data Overview
data link: https://drive.google.com/drive/folders/1Xl6VU3VXGa4c1A3DRhh5Hfp3ItK4aTbV?usp=sharing

chunks link: https://drive.google.com/drive/folders/1QQvKQPpBcut-Tz-3nQjlDj9cUshg6R0V?usp=sharing

![f265bfc0768daa012a8a3d87762329d](https://github.com/ZCBEZHU/AI4EOFINALERNESTO/assets/160029229/a31094ff-e32f-4b48-beac-a55ced250391)




As illustrated in the figure above, the data is sourced from Sentinel-3, covering predominantly oceanic regions. However, it also includes sea ice and glacier-covered land, which introduces uncertainty into the classification process.


## Project Conclusion
By comparing the classification maps produced by different AI models, it is evident that all models effectively classified the ocean and cloud layers. However, challenges arose in the glacier region at the top of the data source. The AI models struggled to accurately distinguish between glaciers, sea ice, and cloud layers, due to their similar visual characteristics. Nonetheless, for some areas of land, ocean, and clouds, the AI algorithms provided reliable identification. This undoubtedly addresses the issues related to the classification of land（No ice or snow cover）, ocean, and cloud layers.

![16bf8bc2cb82181455bfb96bbd66072](https://github.com/ZCBEZHU/AI4EOFINALERNESTO/assets/160029229/6796bd76-0769-468d-848f-92dc096911e4)


## Preparation process

### Mount Google Drive

```
from google.colab import drive
drive.mount('/content/drive')
nc_dir = '/content/drive/MyDrive/dataset/'
```

### Convert nc datasets to numpy .py file
The script reads .nc files, processes them, saves each one as an image, and finally saves all the processed arrays in a single .npy file.

```
# -*- coding: utf-8 -*-
from netCDF4 import Dataset
import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from pandas import DataFrame


def nc2npy(path):
    nc = Dataset(path)
    print(nc)
    nc = nc.variables[path.split('/')[-1].replace('.nc', '')]
    nc = np.array(nc)
    if 'unc' in path:
        nc[np.where(nc == 255)] = 0
    else:
        nc[np.where(nc == 65535)] = 0
        
    
    print(nc.max(), nc.min(), nc.mean())

    return nc


def main():
    for subset in ["radiance"]:
        res = []
        for i in range(1,22):
            arr = nc2npy("Oa%02d_%s.nc"%(i,subset))
            res.append(arr)
            plt.imsave("Oa%02d_radiance_unc.jpg"%i, arr)
            # plt.show()

        res = np.stack(res, axis=-1)
        print(res.shape)
        np.save("%s.npy"%subset, res)

main()
   
```

### Data preprocessing to extract the eight neighbors of each pixel, and convert to numpy array

```
# Iterate over the height and width of the roi, excluding the border pixels
for i in range(1, roi.shape[0] - 1):
    for j in range(1, roi.shape[1] - 1):
        # Extract a (3, 3, 21) patch centered around the pixel (i, j)
        patch = roi[i-1:i+2, j-1:j+2, :]
        patches.append(patch)
patches_array = np.array(patches)
```

### Trim the mask and generate labels

```
# Trim the mask to exclude boundary labels
trimmed_mask = binary_mask[1:-1, 1:-1]

# Flatten the trimmed mask to get a 1D array of labels
labels = trimmed_mask.flatten()
```

### split and balance the dataset

```
# Assuming patches_array is your X and labels is your y
x_train, x_test, y_train, y_test = train_test_split(patches_array, labels, test_size=0.1, random_state=42)
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))
# Identifying the class with the smaller count to balance the number in each class, for example, let it be 9396
num_class = 7580 # Adjust it to the amount you get
# Extract indices of both classes
indices_class_0 = np.where(labels == 0)[0]
indices_class_1 = np.where(labels == 1)[0]
# Randomly sample 9396 indices from class 0
sampled_indices_class_0 = np.random.choice(indices_class_0, num_class, replace=False)

# Combine the sampled indices with indices of class 1
combined_indices = np.concatenate([sampled_indices_class_0, indices_class_1])

# Extract the corresponding patches and labels
balanced_patches = patches_array[combined_indices]
balanced_labels = labels[combined_indices]

# Split the balanced dataset into training and testing sets with a 1:9 ratio
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(
    balanced_patches, balanced_labels, test_size=0.1, random_state=42
)
```

### Image Plot

Define a function that uses matplotlib to display an image of each stage of the run.

```
def plot_image(map1, reverse=False):
  fig, ax = plt.subplots()
  cmap = ListedColormap(['white','black'])
  if reverse == True:
      cmap = ListedColormap(['black','white'])
  plt.imshow(map1, cmap=cmap, vmin=0, vmax=1)
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
  plt.axis('off')
```
### Image Save

The image is saved using opencv during runtime, and since only two values 0 and 1 exist in the array, here we multiply it by 255 to generate a black and white binary image. If you need to invert the image, use the np.where function to perform logical operations.

```
cv2.imwrite('/content/drive/MyDrive/dataset/cnn-predict.png', np.where(map1 == 0, 1, 0) * 255)
```


## Convolutional Neural Networks (CNN) Implementation

### Network layer configration

This code defines the path of data processing, importing the tensorflow library as a convolutional neural network framework, the network uses a five-layer architecture, a convolutional layer, a pooling layer, spreading and passing through two fully-connected layers, and finally the classification function.

```
import tensorflow as tf
from tensorflow.keras import layers, models
data_dir = '/content/drive/MyDrive/dataset/'
# Define the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(3,3,21), padding='SAME'))
model.add(layers.MaxPooling2D((2, 2)))
# Add dense layers for classification
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 10 is the number of classes
```

### Network Training

We first compiles the model, specifies the optimizer as Adam and the loss function as sparse_categorical_crossentropy, and evaluates the accuracy. It then uses the training data X_train and y_train, sets the number of iterations to 4, and reserves 10% of the data for validation. Next, use the test data X_test and y_test to evaluate the loss and accuracy of the model and output the test accuracy. Finally, save the trained model to the specified directory.

```
channel = []
for _channel in range(0,21):
    channel.append(_channel)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train[:,:,:,channel], y_train, epochs=4, validation_split=0.1)
loss, accuracy = model.evaluate(X_test[:,:,:,channel], y_test)
#model.fit(X_train, y_train, epochs=10, validation_split=0.1)
#loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')
model.save(data_dir + "/cnn.model")
```

### Network Prediction

```
model = tf.keras.models.load_model('/content/drive/MyDrive/dataset/cnn.model')

# y_pred=model.predict(x_test_all, batch_size = 250)
y_pred=model.predict(x_test, batch_size = 250)
y_pred1 = np.argmax(y_pred,axis = 1)
map1=y_pred1.reshape(chunk.shape[0]-2, chunk.shape[1]-2)
plot_image(map1)
cv2.imwrite('/content/drive/MyDrive/dataset/cnn-predict.png', np.where(map1 == 0, 1, 0) * 255)
```

<p align="center">
  <img src="img/cnn-predict.png" style="height: 100vh;">
  <br>
  <em>CNN Prediction Result</em>
</p>

## Random Forest Implementation

In several later sections, we use the sci-kit learn library to implement to correlation models where it is more convenient to use spread data. So here, we spread the shape of a single piece of data from (3,3,21) to (189,) .

```
# (x, 3, 3, 21) -> (x, 9, 21)
X_train = X_train.reshape(-1, 9, 21)
X_test = X_test.reshape(-1, 9, 21)

# (x, 9, 21) -> (x, 9, 189)
X_train = X_train.reshape(-1, 189)
X_test = X_test.reshape(-1, 189)
```

In this section, we import the relevant libraries and then generate classifiers using RandomForestClassifier, specifying the use of 100 subclassifiers.

```
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sm
import pickle
import numpy as np
# Initialise the model with n_estimators specifying the number of trees in the forest
clf = RandomForestClassifier(n_estimators=100)

# We need to reshape the data in order to be compatible with Random Forest
X_reshaped = np.reshape(X_train, (X_train.shape[0], -1))
# Fit the model to your training data
clf.fit(X_reshaped, y_train)

# Predict the classes of the test data
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], -1))
y_pred = clf.predict(X_test_reshaped)
print(sm.classification_report(y_test, y_pred))
pickle.dump(clf, open('/content/drive/MyDrive/dataset/rf_model.pkl', 'wb'))
```

Load the model, change the shape of the data, use the model to classify the pixels, and finally draw the image.

```
model = pickle.load(open('/content/drive/MyDrive/dataset/rf_model.pkl', 'rb'))
x_test_rf = np.reshape(x_test_21, (x_test.shape[0], -1))
print(x_test_rf.shape)
y_pred1 = model.predict(x_test_rf)
print(y_pred1.shape)
map1=y_pred1.reshape(chunk.shape[0]-2, chunk.shape[1]-2)
plot_image(map1)
```

<p align="center">
  <img src="img/rf-predict.png" style="height: 80vh;">
  <br>
  <em>RF Prediction Result</em>
</p>


## Gaussian Implementation

A plain Bayesian based Gaussian classifier was created using sk-learn, trained, and training metrics were shown. Subsequently save the model to Google Drive using pickle.

```
# Initialise Gaussian
gnb = GaussianNB()

# Fit the model to your training data
gnb.fit(X_train, y_train)

# Predict the classes of the test data
y_pred = gnb.predict(X_test)
print(sm.classification_report(y_test, y_pred))
pickle.dump(gnb, open('/content/drive/MyDrive/dataset/gaussian_model.pkl', 'wb'))
```

The image generation steps are similar to the above, and the final generated image is as follows.

<p align="center">
  <img src="img/gaussian-predict.png" style="height: 80vh;">
  <br>
  <em>Gaussian Prediction Result</em>
</p>

## GMM Implementation

Similar to the code for the Gaussian model, the GMM model uses sklean to create and set the categorization category to two, followed by training and model saving.

```
from sklearn.mixture import GaussianMixture
import sklearn.metrics as sm
from sklearn.preprocessing import LabelEncoder

# Initialise Gaussian
gmm = GaussianMixture(n_components=2, random_state=42)

# Fit the model to your training data
gmm.fit(X_train, y_train)

# Predict the classes of the test data
y_pred = gmm.predict(X_test)

print(sm.classification_report(y_test, y_pred))
pickle.dump(gmm, open('/content/drive/MyDrive/dataset/gmm_model.pkl', 'wb'))
```
In prediction section, since using the whole image leads to OOM, predictions are made for each of the five parts of the image, one of which is shown here.

```
for i in range(1, 5):
  x_test = []
  chunk = np.load(os.path.join('/content/drive/MyDrive/dataset','chunk_' + str(i) +'.npy'))
  for i in range(1, chunk.shape[0] - 1):
    for j in range(1, chunk.shape[1] - 1):
        # Extract a (3, 3, 21) patch centered around the pixel (i, j)
        patch = chunk[i-1:i+2, j-1:j+2, :]
        x_test.append(patch)
  x_test = np.array(x_test)
  x_test = np.reshape(x_test, (x_test.shape[0], -1))
  print(x_test.shape)
  y_pred1 = model.predict(x_test)
  print(y_pred1.shape)
  map1=y_pred1.reshape(chunk.shape[0]-2, chunk.shape[1]-2)
  plot_image(map1, reverse=True)
  cv2.imwrite('/content/drive/MyDrive/dataset/gmm-predict-' + str(i) + '.png', map1 * 255)
```

<p align="center">
  <img src="img/gmm-predict.png" style="height: 80vh;">
  <br>
  <em>GMM Prediction Result</em>
</p>

### use the k-means model to predict the image and display the result

```
model = pickle.load(open('/content/drive/MyDrive/dataset/kmeans.pkl', 'rb'))
x_test_rf = np.reshape(x_test_21, (x_test.shape[0], -1))
print(x_test_rf.shape)
y_pred1 = model.predict(x_test_rf)
map1=y_pred1.reshape(chunk.shape[0]-2, chunk.shape[1]-2)
plot_image(map1, reverse = True)
```

<p align="center">
  <img src="img/kmeans-predict.png" style="height: 100vh;">
  <br>
  <em>kmeans Prediction Result</em>
</p>


## VIT Implementation

### load data from npy files into memory
```
data_dir = '/content/drive/MyDrive/dataset'
X_train = np.load(os.path.join(data_dir,'X_train_balanced.npy'))
X_test = np.load(os.path.join(data_dir, 'X_test_balanced.npy'))
y_train = np.load(os.path.join(data_dir,'y_train_balanced.npy'))
y_test = np.load(os.path.join(data_dir,'y_test_balanced.npy'))
```
### This code sets up and trains a Vision Transformer (ViT) model for image classification using TensorFlow and Keras. It employs TensorFlow Addons for the AdamW optimizer.
```
# Install packages needed
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = more_data(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=30,
        epochs=5,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history
```
### Defines various hyperparameters and configurations required for setting up and training the Vision Transformer (ViT) model
```
num_classes = 2 #Can be changed to multi-classed classification
input_shape = (3, 3, 21)#depends on the size of the image we want

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
image_size = 72
patch_size = 6
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 8
mlp_head_units = [2048, 1024]
```
### Data augmentation
```
more_data = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="more_data",
)
more_data.layers[0].adapt(X_train)
```
### Save the model
Here vit is your trained model after the training
```
vit = create_vit_classifier()

history = run_experiment(vit)

vit.save(data_dir + 'vit.model')
```
```
Epoch 1/5
4468/4468 [==============================] - 362s 79ms/step - loss: 0.0882 - accuracy: 0.9933 - top-5-accuracy: 1.0000 - val_loss: 0.0128 - val_accuracy: 0.9981 - val_top-5-accuracy: 1.0000
Epoch 2/5
4468/4468 [==============================] - 356s 80ms/step - loss: 0.0227 - accuracy: 0.9956 - top-5-accuracy: 1.0000 - val_loss: 0.0067 - val_accuracy: 0.9983 - val_top-5-accuracy: 1.0000
Epoch 3/5
4468/4468 [==============================] - 355s 80ms/step - loss: 0.0168 - accuracy: 0.9961 - top-5-accuracy: 1.0000 - val_loss: 0.0119 - val_accuracy: 0.9983 - val_top-5-accuracy: 1.0000
Epoch 4/5
4468/4468 [==============================] - 354s 79ms/step - loss: 0.0135 - accuracy: 0.9969 - top-5-accuracy: 1.0000 - val_loss: 0.0181 - val_accuracy: 0.9973 - val_top-5-accuracy: 1.0000
Epoch 5/5
4468/4468 [==============================] - 346s 77ms/step - loss: 0.0141 - accuracy: 0.9963 - top-5-accuracy: 1.0000 - val_loss: 0.0080 - val_accuracy: 0.9986 - val_top-5-accuracy: 1.0000
518/518 [==============================] - 16s 31ms/step - loss: 0.0134 - accuracy: 0.9979 - top-5-accuracy: 1.0000
Test accuracy: 99.79%
Test top 5 accuracy: 100.0%
successfully loaded
```
### load the model and plot the predict image
```
import tensorflow as tf
import tensorflow_addons as tfa

optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
model = tf.keras.models.load_model('/content/drive/MyDrive/dataset/vit.model', custom_objects={'Addons>AdamW': tfa.optimizers.AdamW})

# y_pred=model.predict(x_test_all, batch_size = 250)
y_pred=model.predict(x_test_21, batch_size = 250)
y_pred1 = np.argmax(y_pred,axis = 1)
map1=y_pred1.reshape(chunk.shape[0]-2, chunk.shape[1]-2)
plot_image(map1)
```
<p align="center">
  <img src="img/vit-predict.png" style="height: 100vh;">
  <br>
  <em>VIT Prediction Result</em>
</p>
