from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('best_model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    
    input_data = np.array([int(data1), int(data2), int(data3), int(data4),int(data5),int(data6),int(data7),int(data8),None,None,None,None,None,None,None,None])
    input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = model.predict(input_data_reshaped)
    print(prediction)
    return "Predicted House price is "+str(prediction[0])
    
    

if __name__ == "__main__":
    app.run(debug=True)














