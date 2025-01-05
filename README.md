import cv2
import numpy as np
import glob
import imutils

template = cv2.imread("ppp.png", 0)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]

for imagePath in glob.glob("images/*.jpg"):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None

    for scale in np.linspace(0.5, 1.5, 20)[::-1]:
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    print(found)
    if found is not None and found[0] > 0.06:  # 設置匹配閾值
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        # 畫框並保存區域
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        plate = image[startY:endY, startX:endX]
        cv2.imwrite(f"output/plate_{imagePath.split('/')[-1]}", plate)

        cv2.imshow("Detected Plate", plate)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
