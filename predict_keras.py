import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.map_fn import _result_batchable_to_flat


origin = pd.read_csv("recipe_test.csv")

n_steps = pd.Series(np.array(origin["n_steps"]).astype(int))
n_ingr = pd.Series(np.array(origin["n_ingredients"]).astype(int))

name = pd.read_csv('recipe_text_features_doc2vec100/test_name_doc2vec100.csv', header=None)
ingr = pd.read_csv('recipe_text_features_doc2vec100/test_ingr_doc2vec100.csv', header=None)
steps = pd.read_csv('recipe_text_features_doc2vec100/test_steps_doc2vec100.csv', header=None)

data = pd.concat([name,ingr, steps, n_steps, n_ingr], axis=1)

model = tf.keras.models.load_model("doc2vec100_without_nums")

result_vectors = model.predict(data)

result = []

with open("result.csv", "w") as f:
    f.write("id,duration_label\n")
    for i in range(len(result_vectors)):
        f.write("{},{}\n".format(i+1, float( np.argmax(result_vectors[i]))+ 1) ) 


#print(result)