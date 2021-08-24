from flask import Flask

server = Flask(__name__)


@server.before_first_request
def before_first_request():
    print('test')


@server.route("/")
def hello_world():
    return "Howdy, partner!"
