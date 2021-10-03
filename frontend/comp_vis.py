import cv2
import numpy as np


class ArgusCV():
    def draw_text(img, text, font=cv2.FONT_HERSHEY_PLAIN, pos=(0, 0), font_scale=3, font_thickness=2, text_color=(0, 255, 0), text_color_bg=(0, 0, 0)):
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
        cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
        return text_size

    def create_heatmap(im_map, im_cloud, kernel_size=(5,5),colormap=cv2.COLORMAP_JET,a1=0.5,a2=0.5):
        im_cloud_blur = cv2.GaussianBlur(im_cloud,kernel_size,0)
        im_cloud_clr = cv2.applyColorMap(im_cloud_blur, colormap)
        return (a1*im_map + a2*im_cloud_clr).astype(np.uint8) 