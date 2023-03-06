from sklearn import utils
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/calcular', methods=['POST'])
def calcular():
  modelo_carregado = joblib.load('modelo.pkl')
  dados = request.json
  X_test = pd.DataFrame({
      'CreditScore':[dados['CreditScore']],
      'Geography':[dados['Geography']],
      'Gender':[dados['Gender']],
      'Age':[dados['Age']],
      'Tenure':[dados['Tenure']],
      'Balance':[dados['Balance']],
      'NumOfProducts':[dados['NumOfProducts']],
      'HasCrCard':[dados['HasCrCard']],
      'IsActiveMember':[dados['IsActiveMember']],
      'EstimatedSalary':[dados['EstimatedSalary']]
  })
  y_pred = modelo_carregado.predict_proba(X_test)
  value = y_pred[0][0]*100
  resultado = {"result": float(value)}
  return jsonify(resultado)


if __name__ == '__main__':
  app.run('0.0.0.0')
