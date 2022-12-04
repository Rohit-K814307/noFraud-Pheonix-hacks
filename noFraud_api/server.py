from flask import Flask
from flask_cors import CORS
from flask import Flask, jsonify
import tensorflow as tf
tf.gfile = tf.io.gfile
import tensorflow_hub as hub
from nltk.tokenize import word_tokenize
import tensorflow_text
import re
import numpy as np


app = Flask(__name__)
CORS(app)


def clean_data(line):
    line = str(line)
    line = line.lower() #makes it lowercase

    line = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-\\]", "", line) #takes out any symbols

    tokens = word_tokenize(line)

    words = [word for word in tokens if word.isalpha()] #check if only letters (no special chars/symbols)

 
    return ','.join(words)

def model(metrics):

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    
    encoder_inputs = preprocessor(text_input)
    encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
        trainable=True)
    
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"] 
    sequence_output = outputs["sequence_output"]

    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(32, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(4, activation='softmax')(net)
    
    model = tf.keras.models.Model(inputs=text_input, outputs=out)
    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=metrics)
    
    return model


METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

noFraud_model = model(METRICS)
noFraud_model.load_weights("noFraud_model/phoenix_hacks_model")

@app.route("/")
def home():
    return jsonify({"Welcome to the noFraud API!":"This is the home page"})


@app.route("/predict/<string:text>")
def modelpredict(text:str):
    pred = noFraud_model.predict([clean_data(text)])
    indToCond = {0:"benign",1:"fraud",2:"phishing",3:"spam"}

    return jsonify({"Pred":f'Your email is likely: {indToCond.get(np.argmax(pred))} with a confidence of {pred.max() * 100}%'})



if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port='9999',use_reloader=False)