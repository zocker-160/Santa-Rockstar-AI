#!/usr/bin/env python3 

import os
import time
import asyncio

import pyautogui as pa

import keyboard # needs root on Linux 

from mss import mss
from mss.base import MSSBase, ScreenShot
from python_imagesearch.imagesearch import imagesearch, imagesearch_loop

from PIL import Image

import json
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, Sequential

model: Sequential

# load reference data
ref1Name = "images/img_x1.png"

#refImage1 = Image.open(ref1Name)

t1 = time.time()
#refPoint = pa.locateCenterOnScreen("img_x1.png", grayscale=True)
#refPoint = imagesearch(ref1Name, precision=0.8)
refPoint = imagesearch_loop(ref1Name, timesample=0.5, precision=0.8)
t2 = time.time()

#print("time taken:", (t2-t1)*1000)

print(refPoint)

xRef, yRef = refPoint
yRef -= 25

refLines = [
    ( xRef - 680, yRef),
    ( xRef - 550, yRef),
    ( xRef - 425, yRef),
    ( xRef - 290, yRef),
    ( xRef - 155, yRef),
]

line1 = 370,897
line2 = 497,900
line3 = 625,898
line4 = 756,902
line5 = 885,899

TIME_SLEEP = 0

pa.moveTo((xRef, yRef))
time.sleep(TIME_SLEEP)

for r in refLines:
    print(r)
    #pa.moveTo(r)
    #time.sleep(TIME_SLEEP)

pa.moveTo(refLines[0])
pa.click()

#corrList = [0, 120, 135, 135, 135]
#posList = [0, 120, 255, 390, 524]


TIME_SLEEP = 0.05

bbox = ( refLines[0][0], yRef, refLines[4][0], yRef+16 )
print(bbox)

KEYMAPPING = ["s", "d", "f", "h", "j", "."]

POSLIST = [0, 130, 255]
#LIMITLIST = [130, 140, 190]

LIMITLIST = [150, 200, 200]

#CATLIST = ["green", "red", "yellow", "znone"]

with open("trainset/classes.json", "r") as j:
    CATLIST = json.load(fp=j)

resultList = list()

async def _saveImage(data: ScreenShot, color: str):
    Image.frombytes(
        "RGB", 
        data.size, 
        data.bgra, "raw", "BGRX"
        ).save(os.path.join("trainset", color, f"{time.time()}.png"))

async def _triggerButton(button: int):

    #await asyncio.sleep(0.02)
    keyboard.press(KEYMAPPING[button])
    await asyncio.sleep(TIME_SLEEP)
    keyboard.release(KEYMAPPING[button])

    #await asyncio.sleep(TIME_SLEEP + 0.115)
    await asyncio.sleep(TIME_SLEEP)


async def _checkAI(data: ScreenShot):
    img = Image.frombytes("RGB", data.size, data.bgra, "raw", "BGRX")
    input_arr = keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])

    #print(input_arr)

    t1 = time.time()
    prediction = model.predict(input_arr)
    score = tf.nn.softmax(prediction[0])    
    t2 = time.time()

    pred_class = np.argmax(score)
    confidence = round(np.max(score)*100, 2)
    time_taken = round(t2-t1, 2)

    if pred_class < 3:
        print(f"trigger {pred_class}; confidence: {confidence}%; prediction time: {time_taken} ms")
        #await _triggerButton(pred_cath)

        triggerList = [ _triggerButton(i) for i, sc in enumerate(score) if sc.numpy() > 0.05 and i < len(CATLIST)-1]
        await asyncio.gather( *triggerList )
        #print(score)

        global resultList
        resultList.append(confidence)

        if pred_class == 0 or pred_class == 1:
            #await _saveImage(data, CATLIST[pred_class])
            pass
    else:
        if random.randint(0, 1000) <= 1 and confidence > 0.50:
            #await _saveImage(data, CATLIST[-1])
            #print("saved none image", CATLIST[-1])
            pass

async def _loop(sct: MSSBase):
    await _triggerButton(-1)

    # main loop
    while True:
        pxData = sct.grab(bbox)

        #t1 = time.time()
        await _checkAI(pxData)
        #t2 = time.time()

        if keyboard.is_pressed(KEYMAPPING[3]):
            await _saveImage(pxData, CATLIST[3])

        #await asyncio.sleep(TIME_SLEEP)
        #await asyncio.gather( *[ _check(pxData, i) for i, _ in enumerate(POSLIST) ] )


if __name__ == "__main__":

    model = load_model("AIGen6.2.h5")

    loop = asyncio.get_event_loop()

    with mss() as sct:

        loop.create_task(_loop(sct))

        try:
            #_loop(sct)
            loop.run_forever()
        except:
            print("exit...")
        finally:
            print("lowest confidence:", min(resultList))
            #print("number under 200:", len([ i for i in resultList if i < 200 ]))

            exit()
