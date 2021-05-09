from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from utils import all_diseases, all_symptoms, data_to_index, symptoms_to_disease
from wtforms import Form, TextField, validators, SubmitField, IntegerField

class ReusableForm(Form):
    number_symptoms = IntegerField("Enter the number of symptoms you wish to input:", validators = [validators.InputRequired(), validators.NumberRange(min = 1, max = 4, message = 'You must have at least one and no more than 4 symptoms.')])
    symptoms = TextField("Enter your symptoms below. Ensure that each symptom is seperated by a single space, and multi-word symptoms are seperated with underscores.", validators = [validators.InputRequired()])

    submit = SubmitField("Enter")

def load_keras_model():
    global model
    model = load_model('model=010.h5')

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    #Home page with form
    form = ReusableForm(request.form)

    if request.method == 'POST' and form.validate():
        n_symptoms = request.form['number_symptoms']
        symptoms = request.form['symptoms']
        symptoms = symptoms.split()
        disease = symptoms_to_disease(symptoms, model)
        return render_template('output.html', input = disease)


    return render_template('main.html', form = form)


def main():
    return "<h1>YEEEET</h1>"

if __name__ == '__main__':

    load_keras_model()
    app.run()
