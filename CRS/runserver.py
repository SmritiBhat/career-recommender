"""
This script runs the Foodology application using a development server.
"""

from os import environ
from CRS import app

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '8000'))
    except ValueError:
        PORT = 8008
    app.run(HOST, PORT)
