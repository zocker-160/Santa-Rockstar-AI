#!/usr/bin/env python3 

from sys import intern
import time
import asyncio
from numpy.core.multiarray import result_type

import pyautogui as pa

import keyboard

from mss import mss
from mss.base import MSSBase, ScreenShot
from python_imagesearch.imagesearch import imagesearch, imagesearch_loop

from PIL import Image

# load reference data
ref1Name = "img_x1.png"

#refImage1 = Image.open(ref1Name)

t1 = time.time()
#refPoint_1 = pa.locateCenterOnScreen("img_x1.png", grayscale=True)
#refPoint_1 = imagesearch(ref1Name, precision=0.8)
refPoint_1 = imagesearch_loop(ref1Name, timesample=0.5, precision=0.8)
t2 = time.time()

print("time taken:", (t2-t1)*1000)

print(refPoint_1)

xRef, yRef = refPoint_1
yRef -= 25

print(xRef)

Ref1 = ( xRef - 680, yRef)
Ref2 = ( Ref1[0] + 120, yRef)
Ref3 = ( Ref2[0] + 135, yRef)
Ref4 = ( Ref3[0] + 135, yRef)
Ref5 = ( Ref4[0] + 135, yRef)

print(Ref1)
print(Ref2)
print(Ref3)
print(Ref4)
print(Ref5)

line1 = 370,897
line2 = 497,900
line3 = 625,898
line4 = 756,902
line5 = 885,899

TIME_SLEEP = 0

pa.moveTo((xRef, yRef))
time.sleep(TIME_SLEEP)
pa.moveTo(Ref1)
time.sleep(TIME_SLEEP)
pa.moveTo(Ref2)
time.sleep(TIME_SLEEP)
pa.moveTo(Ref3)
time.sleep(TIME_SLEEP)
pa.moveTo(Ref4)
time.sleep(TIME_SLEEP)
pa.moveTo(Ref5)

pa.moveTo(Ref1)
pa.click()

TIME_SLEEP = 0.05

bbox = ( Ref1[0], yRef, Ref5[0], yRef+1 )

print(bbox)

KEYMAPPING = ["s", "d", "f", "h", "j", "."]

corrList = [0, 120, 135, 135, 135]
posList = [0, 120, 255, 390, 524]

POSLIST = [0, 120, 255]
LIMITLIST = [130, 140, 190]

resultList = list()

async def _triggerButton(button: int):

    await asyncio.sleep(0.02)
    keyboard.press(KEYMAPPING[button])
    await asyncio.sleep(TIME_SLEEP)
    keyboard.release(KEYMAPPING[button])

    await asyncio.sleep(TIME_SLEEP + 0.115)


async def _check(data: ScreenShot, iter: int):
    pix = data.pixel(POSLIST[iter], 0)

    trigger = False

    if iter == 0: # green
        if pix[1] > LIMITLIST[iter] and pix[1] == max(pix):
            trigger = True
        elif pix[1] > LIMITLIST[iter] + 30:
            trigger = True
            
    elif iter == 1: # red
        if pix[0] > LIMITLIST[iter] and pix[0] == max(pix):
            trigger = True
        elif pix[0] > LIMITLIST[iter] + 30:
            trigger = True

    elif iter == 2: # yellow
        if any(x > LIMITLIST[iter] for x in pix):
            trigger = True
    else:
        return
    
    if trigger:
    #if any(x > LIMITLIST[iter] for x in pix):
        await _triggerButton(iter)
        global resultList
        print(KEYMAPPING[iter], pix)
        resultList.append(max(pix))


async def _loop(sct: MSSBase):
    await _triggerButton(-1)

    # main loop
    while True:
        pxData = sct.grab(bbox)

        img = Image.frombytes("RGB", pxData.size, pxData.bgra, "raw", "BGRX")
        img.save("test.png")

        exit()

        await asyncio.gather( *[ _check(pxData, i) for i, _ in enumerate(POSLIST) ] )

        """
        for i, pl in enumerate(POSLIST):
            pix = pxData.pixel(pl, 0)
            #summ = sum( pix )
            #if summ > 550:
            if any( x > 170 for x in pix):
                #print(summ)
                #print(max(pix))

                await _triggerButton(i)
                print(KEYMAPPING[i], max(pix))
                #print(f"pressing {KEYMAPPING[i]}; confidence: { (max(pix) / 255) * 100 }% \n")
                #break
        """
        continue

        colorList = list()
        for i in range(5):
            colorList.append( pxData.pixel(posList[i], 0) )

        print(colorList)
        print([ sum(c) for c in colorList ])

        time.sleep(1)


if __name__ == "__main__":

    loop = asyncio.get_event_loop()

    with mss() as sct:

        loop.create_task(_loop(sct))

        try:
            #_loop(sct)
            loop.run_forever()
        except:
            print("exit...")
        finally:
            print("lowest value:", min(resultList))
            print("number under 200:", len([ i for i in resultList if i < 200 ]))

            exit()
