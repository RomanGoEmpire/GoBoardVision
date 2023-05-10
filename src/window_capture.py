import cv2
import numpy as np
import win32con
import win32gui
import win32ui


class WindowsCapture:

    @staticmethod
    def get_screenshot():
        # define your monitor width and height
        w, h = 1920, 1080

        # for now we will set hwnd to None to capture the primary monitor
        # hwnd = win32gui.FindWindow(None, window_name)
        hwnd = None

        # get the window image data
        w_dc = win32gui.GetWindowDC(hwnd)
        dc_obj = win32ui.CreateDCFromHandle(w_dc)
        c_dc = dc_obj.CreateCompatibleDC()
        data_bit_map = win32ui.CreateBitmap()
        data_bit_map.CreateCompatibleBitmap(dc_obj, w, h)
        c_dc.SelectObject(data_bit_map)
        c_dc.BitBlt((0, 0), (w, h), dc_obj, (0, 0), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        signed_ints_array = data_bit_map.GetBitmapBits(True)
        img = np.fromstring(signed_ints_array, dtype='uint8')
        img.shape = (h, w, 4)

        # free resources
        dc_obj.DeleteDC()
        c_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, w_dc)
        win32gui.DeleteObject(data_bit_map.GetHandle())

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
