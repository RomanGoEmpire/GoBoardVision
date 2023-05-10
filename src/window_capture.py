from ctypes import windll

import cv2
import numpy as np
import win32con
import win32gui
import win32ui
from PIL.Image import Image
from mss import mss


class WindowsCapture:

    @staticmethod
    def get_screenshot():
        # define your monitor width and height
        w, h = 1920, 1080

        # for now we will set hwnd to None to capture the primary monitor
        # hwnd = win32gui.FindWindow(None, window_name)
        hwnd = None

        # get the window image data
        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (w, h), dcObj, (0, 0), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (h, w, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # drop the alpha channel to work with cv.matchTemplate()
        img = img[..., :3]

        # make image C_CONTIGUOUS to avoid errors with cv.rectangle()
        img = np.ascontiguousarray(img)
        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        return img

    def get_window_names(self):
        window_names = []

        def callback(hwnd, _):
            window_names.append(win32gui.GetWindowText(hwnd))
            return True

        win32gui.EnumWindows(callback, None)
        return window_names
