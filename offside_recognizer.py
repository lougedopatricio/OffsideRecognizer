import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import math
from line import Line

NMS_THRESHOLD=0.3
MIN_CONFIDENCE=0.2

def lines_to_list(lines):
    list = []
    for i in range(0,len(lines)):
        for	j in range(0, len(lines[i])):
            p1 = (lines[i][j][0], lines[i][j][1])
            p2 = (lines[i][j][2], lines[i][j][3])
            list.append(Line(p1, p2))
    return list

def get_biggest_lines(lines, number=2):
    number = 2 if number < 1 else number
    
    ordered = sorted(lines, key= lambda l : l.len) #order the lines with respect to their size
    
    if (len(ordered) <= number):
        return ordered
    return ordered[0:number]


def get_vertical_lines(lines): #Angles should be expresed in radians
	return get_lines_oriented(lines, 1.3, 2.53073)

def get_lines_oriented(lines, min_angle, max_angle): #Angles should be expresed in radians
	return [l for l in lines if min_angle <= l.angle <= max_angle or min_angle+math.pi/2 <= l.angle <= max_angle+math.pi/2]


def draw_lines_simple(lines,image):

	if lines is not None:
		for l in lines:
			cv2.line(image, l.p1, l.p2, (0,255,0), 3, cv2.LINE_AA)


		'''
		for i in range(0, len(lines)):

			rho = lines[i][0]
			theta = lines[i][1]
			a = math.cos(theta)
			b = math.sin(theta)
			x0 = a * rho
			y0 = b * rho
			pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
			pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))


			cv2.line(image, pt1, pt2, (0,255,0), 3, cv2.LINE_AA)

		'''
def draw_lines_show_image(auxiliar_lines, attacker_line, defender_line, decision, image):
    red = (5,5,255)
    green = (20, 255, 110)
    blue = (255, 100, 15)
    
    #Draw auxiliar lines
    for line in auxiliar_lines:
        rho = line[0]
        theta = line[1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        
    #Draw defender line
    rho = defender_line[0]
    theta = defender_line[1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(image, pt1, pt2, blue, 3, cv2.LINE_AA)
    
    #Draw attacker line
    rho = attacker_line[0]
    theta = attacker_line[1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    if decision:
        cv2.line(image, pt1, pt2, red, 3, cv2.LINE_AA) # red because is offsede
    else:
        cv2.line(image, pt1, pt2, green, 3, cv2.LINE_AA) # green because is not offside
     
    cv2.imshow(image, 0)
    
    return None



def pedestrian_detection(image, model, layer_name, personidz=0):
	(H, W) = image.shape[:2]
	results = []


	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	model.setInput(blob)
	layerOutputs = model.forward(layer_name)

	boxes = []
	centroids = []
	confidences = []

	for output in layerOutputs:
		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personidz and confidence > MIN_CONFIDENCE:

				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
	# ensure at least one detection exists
	if len(idzs) > 0:
		# loop over the indexes we are keeping
		for i in idzs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
			res = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(res)
	# return the list of results
	return results


def rectangle_mask(shape, vertex1, vertex2):
    rectangle = np.zeros(shape[:2], dtype="uint8")
    cv2.rectangle(rectangle, vertex1, vertex2, 255, -1)
    return rectangle


#------------------------------------------------------------------
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"
model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
#------------------------------------------------------------------


image = cv2.imread("example.png")
greyImage = cv2.imread("example.png",cv2.IMREAD_GRAYSCALE)
dst = cv2.Canny(greyImage, 50, 200, None, 3)
#plt.imshow(greyImage)
#plt.show()

#Detect the lines

lines = cv2.HoughLinesP(dst, 1, math.pi / 180, 50, None, 200, 3)
#lines = cv2.HoughLines(greyImage, 1, math.pi / 180, 50, None, 0, 0)

############################## Magic box#######################################
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"

model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_name = model.getLayerNames()
layer_name = [layer_name[i[0] - 1] for i in model.getUnconnectedOutLayers()]
############################## Magic box#######################################
results = pedestrian_detection(image, model, layer_name,
		personidz=LABELS.index("person"))

#draw rectangles
for res in results:
	cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)

mask = np.zeros(image.shape[:2], dtype="uint8")

#Get the two biggest vertical lines
lines_list = lines_to_list(lines)
auxiliar_lines = get_biggest_lines(get_vertical_lines(lines_list), number = 150)
draw_lines_simple(auxiliar_lines,image)

#cv2.imshow('imagen', image)
#cv2.waitKey(0)
plt.imshow(image)
plt.show()