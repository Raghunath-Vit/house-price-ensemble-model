from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

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
    data9 = request.form['i']
    data10 = request.form['j']
    data11 = request.form['k']
    data12 = request.form['l']
    data13= request.form['m']
    arr = np.array([[int(data1), int(data2), int(data3), int(data4),int(data5),int(data6),int(data7),int(data8),int(data9),int(data10),int(data11),int(data12),int(data13)]])

    input_data_reshaped = arr.reshape(1,-1)

    prediction = model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]== 0):
        return 'The Person does not have a Heart Disease'
    else:
        return  'The Person has Heart Disease'
    

if __name__ == "__main__":
    app.run(debug=True)














