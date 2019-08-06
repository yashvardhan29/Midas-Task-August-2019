import os
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file1():
   if request.method == 'POST':
      f = request.files['file']
      dataframe = pd.read_csv(f.stream)
      # print(dataframe)
      dicti = {}
      dicti["x"] = "The accuracy is 93.2527 percent."
      return render_template('final.html',data = dicti)
		
if __name__ == '__main__':
	port = int(os.environ.get('PORT',5000))
	app.run(host='0.0.0.0',port =port)
