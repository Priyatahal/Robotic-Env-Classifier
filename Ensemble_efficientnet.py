from efficientnet.tfkeras import EfficientNetB0, EfficientNetB5, EfficientNetB7
from keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
#%%

# Memory map the file with the correct shape
X = np.memmap('Img_features.npy', dtype='float32', mode='r', shape=(12000, 200, 200, 3))

#%%
# X = np.load("Img_features.npy")
# X = np.random.randint(12,200,200,3)

# shape = (12,200,200,3)

# X = np.random.random(shape)
selected_data = X[:, np.load("Files/selected_features.npy"),:,:]
# np.save("Files/selected_features",selected_features)
# selected_data = np.load("Files/selected_data.npy")
selected_data = selected_data[:,:,np.load("Files/selected_features.npy"),:]

#%%
# EfficientnetB0
num_classes = 12
lbls = np.load("Files/labls.npy")
one_hot_labels = to_categorical(lbls, num_classes=num_classes)
#%%
efficient_net = EfficientNetB0(
    weights='imagenet',
    input_shape=(selected_data.shape[1], selected_data.shape[2], selected_data.shape[3]),
    include_top=False,
    pooling='max'
)
efficient_net.trainable = False
model_1 = Sequential()
model_1.add(efficient_net)
model_1.add(Dense(units=120, activation='relu'))
model_1.add(Dense(units=120, activation='relu'))
model_1.add(Dense(units=num_classes, activation='softmax'))  # num_classes is the number of classes in your dataset

# Convert the labels to one-hot encoded format


model_1.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=categorical_crossentropy,
    metrics=['accuracy']
)

model_1.summary()

model_1.fit(selected_data,one_hot_labels,epochs=1)


#%%
# EfficientnetB5
effnet = EfficientNetB5(weights=None,
                        include_top=False,
                        input_shape=(selected_data.shape[1], selected_data.shape[2], selected_data.shape[3]))
effnet.trainable = False
def build_model():

    model_2 = Sequential()
    model_2.add(effnet)
    model_2.add(GlobalAveragePooling2D())
    model_2.add(Dropout(0.5))
    model_2.add(Dense(5, activation="relu"))
    model_2.add(Dense(units=num_classes, activation='softmax'))  # num_classes is the number of classes in your dataset
    model_2.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=categorical_crossentropy,
        metrics=['accuracy']
    )

    print(model_2.summary())
    return model_2

# Initialize model
model_2 = build_model()
model_2.fit(selected_data,one_hot_labels,epochs=1)

#%%
#EfficientnetB7
effnet1 = EfficientNetB7(weights=None,
                        include_top=False,
                        input_shape=(selected_data.shape[1], selected_data.shape[2], selected_data.shape[3]))
effnet1.trainable = False

model_3 = tf.keras.Sequential([
        effnet,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=num_classes, activation = 'softmax')
    ])
model_3.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss = categorical_crossentropy, 
        metrics = ['accuracy']
    )

model_3.summary()
model_3.fit(selected_data,one_hot_labels,epochs=1)





#%%

model_1_pred = model_1.predict(selected_data)
model_2_pred = model_2.predict(selected_data)
model_3_pred = model_3.predict(selected_data)
final_preds = np.argmax(model_1_pred + model_2_pred + model_3_pred, axis=1)
print(final_preds)



import tensorflow as tf

# Load the saved model
loaded_model = tf.saved_model.load("EffNet_1-20230615T121459Z-001/EffNet_1")
# model2_pred = loaded_model(data)


