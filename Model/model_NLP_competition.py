import pandas as pd 
import numpy as np
from tqdm import tqdm
dataaf = pd.read_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Natural Language Processing with Disaster Tws\Data\Clean_data_for_disaster.csv")

dataaf.isna().sum()
dataaf.dropna(axis=0, inplace=True)

dataaf.drop("content",axis=1,inplace=True)
dataaf.drop("id",axis=1,inplace=True)

# Label encoder to "label" 
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

dataaf["label"] = label_encoder.fit_transform(dataaf["label"])
class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

dataaf["keyword"] = label_encoder.fit_transform(dataaf["keyword"])
class_mapping2 = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

dataaf["location"] = label_encoder.fit_transform(dataaf["location"])
class_mapping3 = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

print("Class Label Mapping:", class_mapping)




 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import to_categorical

# to make sure there is no float in data
total_rows=7613
with tqdm(total=total_rows) as pbar:
    for i in range (len(dataaf["cleantext"])):
        dataaf.iloc[i,3]=f"{dataaf.iloc[i,0]} {dataaf.iloc[i,1]} {dataaf.iloc[i,3]}"
        dataaf.iloc[i,3]=str(dataaf.iloc[i,3])
        pbar.update(1)
        
from sklearn.utils import shuffle
dataaf = shuffle(dataaf)

x = dataaf["cleantext"]
y = dataaf["label"].values # Convert to a NumPy array
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
sequences = tokenizer.texts_to_sequences(x)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
max_sequence_length = 20 #to control the length of the sequences
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization
from keras.layers import Activation



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.25, random_state=42)


num_classes = len(np.unique(y))# Number of classes

embedding_dim = 4
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))

model.add(LSTM(units=60, kernel_regularizer=l2(0.02)))
model.add(Dropout(0.5))

model.add(Dense(units=num_classes, activation='sigmoid', kernel_regularizer=l2(0.01)))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

y_train_enc = to_categorical(y_train, num_classes=num_classes)
y_test_enc = to_categorical(y_test, num_classes=num_classes)

# Convert data types
X_train = np.asarray(X_train).astype(np.float32)
y_train_enc = np.asarray(y_train_enc).astype(np.int32)
X_test = np.asarray(X_test).astype(np.float32)
y_test_enc = np.asarray(y_test_enc).astype(np.int32)


model.fit(X_train, y_train_enc, batch_size=85, epochs=5)
accuracy = model.evaluate(X_test, y_test_enc, verbose=2)
print("Accuracy:", accuracy)


# saving the model
model.save(r'C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Natural Language Processing with Disaster Tws\Model\MODEL_DISASTER_COMP.h5')

import pickle

file_path = r'C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Natural Language Processing with Disaster Tws\Model/tokenizer_dis_comp.pickle'
with open(file_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)