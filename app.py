from flask import Flask,request
from flask_cors import CORS
from recom import recommend

app= Flask(__name__)
CORS(app)

@app.route("/",methods = ["GET"])
def index():
    try:
        q=request.args.get('q')
        data=recommend(q)
        return {
            'data':data,
            'success':True
        }
    except:
        return {
            'message':'Error',
            'success':False
        }

    
if __name__ == "__main__":
    app.run(debug=True)