# Usage
# python build_dataset_train.py

#import package
from pyimagesearch.iou import compute_iou
from pyimagesearch import config
from bs4 import BeautifulSoup
from imutils import paths
import cv2
import os

for dirPath in (config.POSITVE_PATH, config.NEGATIVE_PATH):
    print(dirPath)
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)


imagePaths = list(paths.list_images(config.ORIG_IMAGES))

totalPositive = 0
totalNegative = 0

for (i, imagePath) in enumerate(imagePaths):
    print('[INFO] processing image {}/{}...'.format(i+1, len(imagePaths)))

    filename = imagePath.split(os.path.sep)[-1]
    filename = filename[:filename.rfind(".")]
    annotPath = os.path.sep.join([config.ORIG_ANNOTS, "{}.xml".format(filename)])

    contents = open(annotPath).read()
    soup = BeautifulSoup(contents, "html.parser")
    gtBoxes = []

    w = int(soup.find("width").string)
    h = int(soup.find("height").string)

    # XML 파일에서 Gt 값 가져옴
    for o in soup.find_all("object"):
        label = o.find("name").string
        xMin = int(o.find("xmin").string)
        yMin = int(o.find("ymin").string)
        xMax = int(o.find("xmax").string)
        yMax = int(o.find("ymax").string)

        xMin = max(0, xMin)
        yMin = max(0, yMin)
        xMax = min(w, xMax)
        yMax = min(h, yMax)

        gtBoxes.append((xMin, yMin, xMax, yMax)) # Gt list

    image = cv2.imread(imagePath)

    # 이미지 Segmentation 진행
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    proposedRects =[]

    for (x,y,w,h) in rects:
        proposedRects.append((x, y, x + w, y + h))

    positiveROIs = 0 # Detection 대상이 존재하는 이미지
    negativeROIs = 0 # Detection 대상이 존재하지 않는 이미지

    for proposedRect in proposedRects[:config.MAX_PROPOSALS]:
        (propStartX, propStartY, propEndX, propEndY) = proposedRect

        for gtBox in gtBoxes:
            iou = compute_iou(gtBox, proposedRect) # iou 계산
            (gtStartX, gtStartY, gtEndX, gtEndY) = gtBox

            roi = None
            outputPath = None

            if iou > 0.7 and positiveROIs <= config.MAX_POSITIVE:

                roi = image[propStartY:propEndY, propStartX:propEndX]
                filename = "{}.png".format(totalPositive)
                outputPath = os.path.sep.join([config.POSITVE_PATH, filename])

                positiveROIs += 1
                totalPositive += 1

            # 만약 제시한 Region 이 gt 내부이면 True 아니면 False
            fullOverlap = propStartX >= gtStartX
            fullOverlap = fullOverlap and propStartY >= gtStartY
            fullOverlap = fullOverlap and propEndX <= gtEndX
            fullOverlap = fullOverlap and propEndY <= gtEndY

            if not fullOverlap and iou < 0.05 and negativeROIs <= config.MAX_NEGATIVE:

                roi = image[propStartY:propEndY, propStartX:propEndX]
                filename = "{}.png".format(totalNegative)
                outputPath = os.path.sep.join([config.NEGATIVE_PATH, filename])

                negativeROIs += 1
                totalNegative += 1

            if roi is not None and outputPath is not None:
                roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(outputPath, roi)