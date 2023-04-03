#!/bin/bash
apt-get update
apt-get install -y libgl1-mesa-dev
apt-get install -y build-essential

curl -sL https://deb.nodesource.com/setup_18.x -o nodesource_setup.sh
bash nodesource_setup.sh 
apt-get install -y nodejs

/usr/bin/python3 -m pip install --upgrade pip

pip install gpustat
pip install flask
pip install flask-socketio
pip install scikit-learn
pip install pandas
pip install opencv-python
pip install mediapipe