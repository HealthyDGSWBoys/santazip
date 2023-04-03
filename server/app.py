from flask import Flask
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO

import DeviceReporter as reporter

app = Flask(__name__)
app.config['SECRET_KEY'] = '1234'
socket_io = SocketIO(app, cors_allowed_origins='*')

@app.route('/')
def hello_world():
    return "Hello Gaemigo Project Home Page!!"

@app.route('/chat')
def chatting():
    return render_template('chat2.html')

@socket_io.on('connect')
def onWsConnect(msg):
    print("con")

@socket_io.on('disconnect')
def disconnect():
    print("dis")

def onDeviceUpdate(info):
    socket_io.emit("device", info)

reporter.reporter.setOnUpdate(onDeviceUpdate)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=60002)