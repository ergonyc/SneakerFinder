from flask import Flask
app = Flask(__name__)

from sneakapp import views
from sneakapp import model_utils