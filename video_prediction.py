import cv2
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model("arthro_multiclass_model.h5")

label_dict = {0:"Electrocautery Tool", 1:"Shaver Tool", 2:"No Tool"}

video_data = cv2.VideoCapture("Video_segments/Shaver+electrocautery1_Electro.mp4")

cur_tool_tracking = 2

found_tool = 2

num_diff = 0

frame_num = 1
ret = True
while(ret):
    ret, frame_data = video_data.read()
    if ret:
        img = cv2.resize(frame_data, (128,128))
        prediction = model.predict(np.expand_dims(img, 0))
        found_tool = np.argmax(prediction)
        if found_tool != cur_tool_tracking:
            cur_tool_tracking = found_tool
    
        print("Frame #" + str(frame_num) + " : " + str(found_tool) + " --- " + str(prediction))
        frame_num += 1

        if found_tool == 0:
            plt.imshow(img)
            plt.show()