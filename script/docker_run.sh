docker run --name santazip -p 6001:60001 -p 6002:60002 -p 6003:60003 --gpus all -it -v $(pwd):/santazip --rm tensorflow/tensorflow:2.11.0-gpu
