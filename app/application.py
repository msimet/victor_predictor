"""
application.py: Run the VictorPredictor Flask app.
"""

from vp import app as application

if __name__ == '__main__':
    application.run(debug=True)
