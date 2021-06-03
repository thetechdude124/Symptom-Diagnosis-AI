import tensorflow as tf
import keras
import numpy as np

all_diseases = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer diseae', 
'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension', 'Migraine', 
'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 
'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 
'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins', 'Hypothyroidism', 
'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis', 'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 
'Urinary tract infection', 'Psoriasis', 'Impetigo']

all_symptoms = ['nan', 'itching', 'skin rash', 'continuous sneezing', 
'shivering', 'stomach pain', 'acidity', 'vomiting', 'indigestion', 
'muscle wasting', 'patches in throat', 'fatigue', 'weight loss', 
'sunken eyes', 'cough', 'headache', 'chest pain', 'back pain', 
'weakness in limbs', 'chills', 'joint pain', 'yellowish skin', 
'constipation', 'pain during bowel movements', 'breathlessness', 
'cramps', 'weight gain', 'mood swings', 'neck pain', 'muscle weakness',
'stiff neck', 'pus filled pimples', 'burning micturition', 
'bladder discomfort', 'high fever', 'nodal skin eruptions', 
'ulcers on tongue', 'loss of appetite', 'restlessness', 'dehydration',
 'dizziness', 'weakness of one body side', 'lethargy', 'nausea', 
 'abdominal pain', 'pain in anal region', 'sweating', 'bruising', 
 'cold hands and feets', 'anxiety', 'knee pain', 'swelling joints', 
 'blackheads', 'foul smell of urine', 'skin peeling', 'blister', 
 'dischromic  patches', 'watering from eyes', 'extra marital contacts', 
 'diarrhoea', 'loss of balance', 'blurred and distorted vision', 
 'altered sensorium', 'dark urine', 'swelling of stomach', 
 'bloody stool', 'obesity', 'hip joint pain', 'movement stiffness', 
 'spinning movements', 'scurring', 'continuous feel of urine', 
 'silver like dusting', 'red sore around nose', 'spotting  urination', 
 'passage of gases', 'irregular sugar level', 'family history', 
 'lack of concentration', 'excessive hunger', 'yellowing of eyes', 
 'distention of abdomen', 'irritation in anus', 'swollen legs', 
 'painful walking', 'small dents in nails', 'yellow crust ooze']

def data_to_index(condition, reference_list):
    condition = str(condition)
    condition = condition.strip(' ')
    condition = condition.replace('_', ' ')
    condition.lower()
    number = reference_list.index(condition)
    return number 

def symptoms_to_disease(symptoms_for_model, model):
    index_list = [data_to_index(symptoms_for_model[i], all_symptoms) for i in range(len(symptoms_for_model))]

    if len(index_list) < 4:

        for i in range (4-(len(index_list))):
            index_list.append(0)
    
    index_list = tf.convert_to_tensor(index_list)

    index_list = tf.reshape(index_list, [1, 4])

    max_value = max(max(model.predict(index_list)))
    index = np.where(model.predict(index_list)[0] == max_value)

    prediction = all_diseases[index[0][0]]
    
    return prediction