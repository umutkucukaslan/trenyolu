import os

import cv2
import numpy as np


class ColorPalette:
    def __init__(self, n_colors, color_box_size=(100, 50), icon=None):
        self.n_colors = n_colors
        self.colors = [None for _ in range(n_colors)]
        self.color_box_size = color_box_size
        self.width = color_box_size[1]
        self.window_size = (color_box_size[0], color_box_size[1] * n_colors)
        self.pallette = (
            np.ones((self.window_size[0], self.window_size[1], 3), dtype=np.uint8) * 255
        )
        self.draw_black_lines()
        self.chosen_bin = None
        self.icon = icon
        if self.icon is None:
            self.icon = np.random.uniform(
                0, 200, (color_box_size[0] // 2, color_box_size[1], 3)
            )
            self.icon[:, 0, :] = 0
            self.icon[:, -1, :] = 0
        else:
            self.icon = cv2.resize(
                self.icon, (color_box_size[1], color_box_size[0] // 2)
            )
            print(self.icon.shape)
            print(self.icon.dtype)

        cv2.namedWindow("pallette")
        cv2.setMouseCallback("pallette", self.handle_click)
        self.refresh()

    def draw_black_lines(self):
        for x in range(0, self.window_size[1], self.width):
            self.pallette[:, x - 1 : x + 1, :] = 0

    def refresh(self):
        self.pallette = (
            np.ones((self.window_size[0], self.window_size[1], 3), dtype=np.uint8) * 255
        )
        # draw colors on pallette
        for color_no, color in enumerate(self.colors):
            if color is not None:
                self.pallette[
                    self.color_box_size[0] // 2 :,
                    self.width * color_no : self.width * (color_no + 1),
                    0,
                ] = color[0]
                self.pallette[
                    self.color_box_size[0] // 2 :,
                    self.width * color_no : self.width * (color_no + 1),
                    1,
                ] = color[1]
                self.pallette[
                    self.color_box_size[0] // 2 :,
                    self.width * color_no : self.width * (color_no + 1),
                    2,
                ] = color[2]
            else:
                self.pallette[
                    self.color_box_size[0] // 2 :,
                    self.width * color_no : self.width * (color_no + 1),
                    :,
                ] = self.icon
        # mark bin if chosen
        if self.chosen_bin:
            self.pallette[
                int(self.color_box_size[0] * 0.2) : int(self.color_box_size[0] * 0.3),
                int(self.width * (self.chosen_bin + 0.4)) : int(
                    self.width * (self.chosen_bin + 0.6)
                ),
                :,
            ] = 0
        self.draw_black_lines()
        cv2.imshow("pallette", self.pallette)

    def set_color(self, color):
        if self.chosen_bin:
            self.colors[self.chosen_bin] = color
        self.refresh()

    def handle_click(self, event, x, y, flags, param):
        # print("clicked on: ", (x, y))

        if event == cv2.EVENT_LBUTTONUP:
            if (
                self.chosen_bin is not None
                and int(self.color_box_size[0] * 0.2)
                <= y
                <= int(self.color_box_size[0] * 0.3)
                and int(self.width * (self.chosen_bin + 0.4))
                <= x
                <= int(self.width * (self.chosen_bin + 0.6))
            ):
                self.colors[self.chosen_bin] = None
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print("double left click on ", (x, y))
            if self.chosen_bin is None:
                self.chosen_bin = x // self.width
            else:
                self.chosen_bin = None
            print(self.chosen_bin)
        self.refresh()


icon = cv2.imread("/Users/umutkucukaslan/Desktop/icon.jpg")
pallette_manager = ColorPalette(n_colors=20, color_box_size=(60, 40), icon=icon)


cv2.waitKey()
cv2.destroyAllWindows()


exit()


def resize_nearest(image, multiplier: int = 4):
    y, x = image.shape[0], image.shape[1]
    return cv2.resize(
        image, (x * multiplier, y * multiplier), interpolation=cv2.INTER_NEAREST
    )


def color_quantization(img, n_colors):
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.0)
    K = n_colors
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    print(ret)
    print("--")
    print(label)
    print("--")
    print(center)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


def process(img):
    img = cv2.medianBlur(img, 5)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    return img


if __name__ == "__main__":
    # image_path = "/Users/umutkucukaslan/Desktop/img.jpeg"
    image_path = "/Users/umutkucukaslan/Desktop/img3.png"
    n_xbins = 40

    image_raw = cv2.imread(image_path)
    image_raw_hsv = cv2.cvtColor(image_raw, cv2.COLOR_BGR2HSV)
    # image_raw_hsv[:,:,1] = 255
    # image_raw_hsv[:,:,2] = 127

    # image_raw = process(image_raw)

    n_ybins = int(image_raw.shape[0] * (n_xbins / image_raw.shape[1]))
    image = cv2.resize(image_raw, (n_xbins, n_ybins), interpolation=cv2.INTER_LINEAR)

    n_colors = 10

    image2 = color_quantization(image_raw_hsv, n_colors)
    image2 = cv2.cvtColor(image2, cv2.COLOR_HSV2BGR)
    image2 = cv2.resize(image2, (n_xbins, n_ybins), interpolation=cv2.INTER_LINEAR)

    new_image_path = os.path.join(os.path.dirname(image_path), "downsampled.png")
    cv2.imwrite(new_image_path, resize_nearest(image, 8))
    cv2.imshow("image", resize_nearest(image, 8))
    cv2.imshow("o", image_raw)
    cv2.imshow("color-quant", resize_nearest(color_quantization(image, n_colors), 8))
    cv2.imshow("color-quant2", resize_nearest(image2, 8))
    cv2.waitKey()
    cv2.destroyAllWindows()
