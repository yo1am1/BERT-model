import keras
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from transformers import BertTokenizer, TFBertModel

# set cuda gpu
print(tf.config.list_physical_devices('GPU'))
# physical_devices = tf.config.list_physical_devices('GPU')
#
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# else:
#     print("No GPU found, using CPU instead.")

# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data()

# Convert the sequences to strings
x_train = [" ".join([str(word) for word in sequence]) for sequence in x_train]
x_test = [" ".join([str(word) for word in sequence]) for sequence in x_test]

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the input sequences
x_train_tokens = tokenizer(x_train, padding=True, truncation=True, return_tensors="tf", max_length=512)
x_test_tokens = tokenizer(x_test, padding=True, truncation=True, return_tensors="tf", max_length=512)

# Load the pre-trained BERT model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Freeze BERT layers
for layer in bert_model.layers:
    layer.trainable = False

# Create a classification head
inputs = Input(shape=(512,), dtype=tf.int32)
bert_output = bert_model(inputs)["last_hidden_state"]
pooled_output = bert_output[:, 0, :]
outputs = Dense(1, activation='sigmoid')(pooled_output)

# Create the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train_tokens["input_ids"], y_train, epochs=1, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
accuracy = model.evaluate(x_test_tokens["input_ids"], y_test)[1]
print(f"Test Accuracy: {accuracy}")
