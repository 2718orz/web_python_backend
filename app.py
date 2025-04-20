from flask import Flask, jsonify
from api.predict import predict_route
from flask_cors import CORS
app = Flask(__name__)
CORS(app,resources={r"/api/*": {"origins": "*"}})
app.register_blueprint(predict_route, url_prefix='/api')

if __name__ == '__main__':
    app.run()