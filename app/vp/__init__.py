"""
vp: the internals of the VictorPredictor app, generating pages using Flask.
"""

from flask import Flask
from vp import views

app = Flask(__name__)
