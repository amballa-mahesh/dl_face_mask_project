from flask import Flask, render_template,url_for,Response
import os
os.chdir('e:/Vscode_files/End_to_End_Ml_project_Facemask')
from src.components.utils import generate_frame

app = Flask(__name__)   

@app.route('/')
def index():    
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frames')
    
   
if __name__ =="__main__":
    app.run(debug=True)