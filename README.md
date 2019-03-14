
# Anki Vector Object Detection
This is a fork of [github.com/unbun/Anki-Vector-AI](https://github.com/unbug/Anki-Vector-AI) but instead of using Google Cloud Services, it uses a **local Inception TensorFlow** model.

# Object detection with Vector
This program is to enable Vector to detect objects with its camera, and tell us what it found. 

We take a photo from Vector's camera, infer labels using the pre-downloaded Inception model, then finally, we turn all the label text into a sentence along with the probability and send to Vector so that Vector can say it out loud.

Admittedly, this is not anywhere near as good as the original. But it also doesn't upload your images and instead runs the inference locally, so there's that.

I have no demo videos.

### Run the code yourself
1. Install [Vector Python SDK](https://developer.anki.com/vector/docs/install-macos.html). You can test the SDK by running any of the example from [anki/vector-python-sdk/examples/tutorials/](https://github.com/anki/vector-python-sdk/tree/master/examples/tutorials) 
2. Clone this project to local. It requires Python 3.6+.
3. Install dependencies
```
pip install -r requirements.txt
```
4. Download the model
```
./download_model.sh
```
5. Make sure your computer and Vector in the same WiFi network. Then run `python3 object_detection.py`.
6. If you are lucky, Vector will start the first object detection, and start speaking the labels of what it sees. It's much more hilarious than the original given the Inception model.

### How it works
Read the code, or checkout the original [README](https://github.com/unbug/Anki-Vector-AI) which is mostly the same.

