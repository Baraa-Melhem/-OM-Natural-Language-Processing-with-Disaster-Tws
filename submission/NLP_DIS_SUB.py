import pandas as pd 
import numpy as np
from tqdm import tqdm
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences 



# recall the model
model1 = load_model(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Natural Language Processing with Disaster Tws\Model\MODEL_DISASTER_COMP.h5")


file_path = r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Natural Language Processing with Disaster Tws\Model\tokenizer_dis_comp.pickle"
with open(file_path, 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)
    

test=pd.read_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Natural Language Processing with Disaster Tws\Data\Clean_test_data_for_disaster.csv")
test.drop("content",axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

test["keyword"] = label_encoder.fit_transform(test["keyword"])
class_mapping2 = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

test["location"] = label_encoder.fit_transform(test["location"])
class_mapping3 = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

total_rows=3263 
with tqdm(total=total_rows) as pbar:
    for i in range (len(test["cleantext"])):
        test.iloc[i,3]=f"{test.iloc[i,1]} {test.iloc[i,2]} {test.iloc[i,3]}"
        test.iloc[i,3]=str(test.iloc[i,3])
        pbar.update(1)






sequences = loaded_tokenizer.texts_to_sequences(test["cleantext"])
max_sequence_length = 20 
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

predictions = model1.predict(padded_sequences)

test.rename(columns={"keyword": 'target'}, inplace=True)
test.drop("location",axis=1,inplace=True)
test.drop("cleantext",axis=1,inplace=True)

total_rows=3263 
with tqdm(total=total_rows) as pbar:
    for i in range (len(test["cleantext"])):
        predictedind1= np.argmax(predictions[i])
        test.iloc[i,1]=predictedind1
        pbar.update(1)
        
test.to_csv(r"C:\Users\User\Desktop\Artificial intelligence\Machine learning\Deep learning\Natural Language Processing with Disaster Tws\Data\NLP_DIS_submission.csv",index=False)

