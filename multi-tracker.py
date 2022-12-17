import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def get_vanishing_point(line_a, line_b) -> tuple(float, float):
    #lines = cv2.HoughLines()
    
    theta_a = line_a[1]
    m_a = 0 - (math.cos(theta_a)/math.sin(theta_a))
    n_a = line_a[0]/math.sin(theta_a)
    
    theta_b = line_b[1]
    m_b = 0 - (math.cos(theta_b)/math.sin(theta_b))
    n_b = line_b[0]/math.sin(theta_b)
    
    A = np.array([[1,-m_a], [1, -m_b]])
    b = np.array([n_a, n_b])
    y,x = np.linalg.solve(A,b)
    
    return x,y

def get_biggest_lines(lines, number=2) -> list(float, float):
    number = 2 if number < 1 else None
    
    ordered = sorted(lines, key= lambda l : l[0]) #order the lines with respect to their size
    
    if (len(ordered) >= number):
        return ordered
    return ordered[0:number]

def get_vertical_lines(lines) -> list(float, float): #Angles should be expresed in radians
    return get_lines_oriented(lines, 0,61, 2,53073)

def get_lines_oriented(lines, min_angle, max_angle) -> list(float, float): #Angles should be expresed in radians
    return [l for l in lines if min_angle <= l[1] <= max_angle]

def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]

def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist

def detect_players(img) -> list():
    raise NotImplementedError("Complete this method!") 


def split_attackers_defenders(players) -> tuple(list(), list()):
    raise NotImplementedError("Complete this method!") 


def get_player_most_close_to_goal(players, vanishing_point):
    closest = None  # ((x,y), line_player)
    for p in players:
        #Create a line which passes through the vanishing point and the player
            #y = (y2-y1)*((x-x1)/(x2-x1)) + y1
            #m = ((y2-y1)/(x2-x1))
            #n = (-1)*x1*m + y2
            
        x1, y1 = vanishing_point[0], vanishing_point[1]
        x2, y2 = p[0], p[1]
        
        m = ((y2-y1)/(x2-x1))
        n = (-1)*x1*m + y2
        
        a = np.array([[1,-m], [1, 1/m]])
        b = np.array([n,0])
        c = np.linalg.solve(a,b) #solve the ecuation system
        x0, y0 = c[1], c[0] #point where the line and it´s perpendicular which passes through the origin met
        
        line_distance = math.sqrt(x0**2 + y0**2)
        
        if (y0>= 0):
            if x0>=0:
                theta = math.atan(y0/x0) #first cuadrant
            else:
                theta = math.pi - math.atan(y0/((-1)*x0)) #second cuadrant 
        else:
            if x0>=0:
                theta = math.pi*2 - math.atan(((-1)*y0)/x0) #fourth cuadrant
            else:
                theta = math.pi + math.atan(((-1)*y0)/((-1)*x0)) #third cuadrant 
                
        line_player = (line_distance, theta)
                
        if closest != None:
            angle_closest = closest[1]
        else:
            angle_closest = -1*np.inf
        
        #convert the angles into the third or fourth cuadrant
        if math.pi/2 <= angle_closest <= math.pi:
            angle_closest = angle_closest + math.pi
        elif 0 <= angle_closest <= math.pi/2:
            angle_closest = angle_closest + math.pi
        
        if math.pi/2 <= theta <= math.pi:
            theta = theta + math.pi
        elif 0 <= theta <= math.pi/2:
            theta = theta + math.pi
            
        #if it is the most close actualize the closets
        if theta>angle_closest or closest in None:
            closest = (p, line_player)
        
        
    return closest


def is_offside(attacker, defender):
    theta_1 = 0 #Offensive player line
    theta_2 = 0 #Defensive player line
    offside = False
    
    '''
    if (math.pi/2 <= theta_1 <= math.pi): # Line 1 on the third cuadrant
        if (math.pi/2 <= theta_2 <= math.pi): # Both lines on the first cuadrant
            if (theta_1 > theta_2):
                offside = True
        else
    else:
        '''
        
    # Convert the angles into the first or fourth cuadrant angles (we´re just considering the right part of the field)
    if (math.pi/2 <= theta_1 <= math.pi): 
        theta_1 = theta_1+math.pi
    elif (0 <= theta_1 <= math.pi/2):
        theta_1 = theta_1+math.pi
        
        
    if (math.pi/2 <= theta_2 <= math.pi): 
        theta_2 = theta_2*math.pi
    elif (0 <= theta_2 <= math.pi/2):
        theta_2 = theta_2+math.pi
    
    if (theta_1 > theta_2):
        offside = True
    
    return offside

def get_attacker_defender_lines(attacker, defender):
    
    return None

def draw_lines_show_image(auxiliar_lines, attacker_line, defender_line, decision, image):
    red = (5,5,255)
    green = (20, 255, 110)
    blue = (255, 100, 15)
    
    #Draw auxiliar lines
    for line in auxiliar_lines:
        rho = line[0][0]
        theta = line[0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        
    #Draw defender line
    rho = defender_line[0][0]
    theta = defender_line[0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(image, pt1, pt2, blue, 3, cv2.LINE_AA)
    
    #Draw attacker line
    rho = defender_line[0][0]
    theta = defender_line[0][1]
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

def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

# plot points
def draw_circle(img, center, radius, color):
    img = cv2.circle(img, (center[0], center[1]), radius, color, -1)

def draw_cross(img, center, color, d):
    cv2.line(img,
             (center[0] - d, center[1] - d), (center[0] + d, center[1] + d),
             color, 1, cv2.LINE_AA, 0)
    cv2.line(img,
             (center[0] + d, center[1] - d), (center[0] - d, center[1] + d),
             color, 1, cv2.LINE_AA, 0)

def distance(pt1, pt2):
    return math.hypot(pt2[0] - pt1[0],pt2[1] - pt1[1])

def find_midpts(pts):
    pt1 = pts[0]
    pt2 = pts[1]
    pt3 = pts[2]
    pt4 = pts[3]

    d12 = distance(pt1, pt2)
    d13 = distance(pt1, pt3)
    d14 = distance(pt1, pt4)

    if max(d12,d13,d14) == d12:
        return ((pt1+pt2)*0.5).astype('int')
    elif max(d12,d13,d14) == d13:
        return ((pt1+pt3)*0.5).astype('int')
    elif max(d12,d13,d14) == d14:
        return ((pt1+pt4)*0.5).astype('int')

#Reference : https://docs.opencv.org/trunk/db/df8/tutorial_py_meanshift.html
def camshift_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    x,y,w,h = detect_one_face(frame)

    # Write track point for first frame
    output.write("%d,%d,%d\n" % (frameCounter, x+w/2, y+h/2)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (x,y,w,h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (x,y,w,h)) # this is provided for you

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)           #Uncomment these for display

        (pt_x,pt_y) = find_midpts(pts)
        draw_cross(frame,(np.int32(pt_x),np.int32(pt_y)), (0, 255, 0), 3)   #Uncomment these for display
        output.write("%d,%d,%d\n" % (frameCounter, np.int32(pt_x), np.int32(pt_y))) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1
        cv2.imshow('frame', frame)                              #Uncomment these for display
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    output.close()

#Reference : http://www.morethantechnical.com/2011/06/17/simple-kalman-filter-for-tracking-using-opencv-2-2-w-code/
def kalman_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
      return

    # detect face in first frame
    x,y,w,h = detect_one_face(frame)

    # Write track point for first frame
    pt = (frameCounter, x+w/2, y+h/2)
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (x,y,w,h)

    # initialize the tracker
    kalman = cv2.KalmanFilter(4,2,0)

    state = np.array([x+w/2,y+h/2,0,0], dtype='float64') # initial position
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                  [0., 1., 0., .1],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        # perform the tracking
        prediction = kalman.predict()

        # generate measurement
        x,y,w,h = detect_one_face(frame)
        if x != 0 and y != 0 :
          measurement = (x+w/2, y+w/2)
          posterior = kalman.correct(measurement)
          draw_cross(frame,(np.int32(posterior[0]),np.int32(posterior[1])), (0, 0, 255), 3)
          cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
          pt = (frameCounter, np.int32(posterior[0]), np.int32(posterior[1]))             #Use posterior
        else:
          draw_cross(frame,(np.int32(prediction[0]),np.int32(prediction[1])), (0, 255, 0), 3)
          pt = (frameCounter, np.int32(prediction[0]), np.int32(prediction[1]))           #Didnt get measurement so using prediction

        # write the result to the output file
        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1    
        cv2.imshow('frame', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    output.close()

def particle_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
      return

    # detect face in first frame
    x,y,w,h = detect_one_face(frame)

    # Write track point for first frame
    pt = (frameCounter, x+w/2, y+h/2)
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (x,y,w,h)

    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (x,y,w,h)) # this is provided for you

    # hist_bp: obtain using cv2.calcBackProject and the HSV histogram
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_bp = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1) 

    #Spread 300 random particles near object to track
    n_particles = 300

    init_pos = np.array([x + w/2.0,y + h/2.0], int) # Initial position
    particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
    # f0 = particleevaluator(hist_bp, pos) * np.ones(n_particles) # Evaluate appearance model
    weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)


    while(1):
        ret,frame = v.read()# read another frame
        if ret == False:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist_bp = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1) 
        # cv2.imshow('hist_bp', hist_bp)

        # perform the tracking
        stepsize = 18;

        # Particle motion model: uniform step (TODO: find a better motion model)
        np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

        # Clip out-of-bounds particles
        particles = particles.clip(np.zeros(2), np.array((frame.shape[1],frame.shape[0]))-1).astype(int)

        f = particleevaluator(hist_bp, particles.T) # Evaluate particles
        
        xrange = range

        #Try to show some visuals
        for i in xrange(len(f)):
            if f[i] >= 1:
                draw_circle(frame, particles[i].T, 1, (0, 0, 255))            #Good Particles
            else:
                draw_circle(frame, particles[i].T, 1, (0, 0, 0))              #Bad Particles

        weights = np.float32(f.clip(1))             # Weight ~ histogram response #clip all bad particles
        weights /= np.sum(weights)                  # Normalize w
        pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average

        draw_cross(frame,(np.int32(pos[0]),np.int32(pos[1])), (0, 255, 0), 3)

        if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
            particles = particles[resample(weights),:]  # Resample particles according to weights

        cv2.imshow('frame', frame)
        pt = (frameCounter, np.int32(pos[0]),np.int32(pos[1]))
        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    output.close()


#Reference : https://docs.opencv.org/3.2.0/d7/d8b/tutorial_py_lucas_kanade.html
def of_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = v.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    x,y,w,h = detect_one_face(old_frame)
    masky = np.zeros_like(old_gray)
    masky[y+5:y+h-5, x+5:x+w-5] = 255

    p0 = cv2.goodFeaturesToTrack(old_gray, mask = masky, **feature_params)

    output.write("%d,%d,%d\n" % (frameCounter, x+w/2, y+h/2)) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(1):
        ret,frame = v.read()
        if ret == False:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            # cv2.arrowedLine(frame, (a,b), (c,d), (255, 0, 0), tipLength=0.5)

        weights = np.ones(good_new.shape[0], dtype='float')
        weights /= np.sum(weights)                  # Normalize w
        pos = np.sum(good_new.T * weights, axis=1).astype(int) # expected position: weighted average
        draw_cross(frame,(np.int32(pos[0]),np.int32(pos[1])), (0, 255, 0), 3)

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

        output.write("%d,%d,%d\n" % (frameCounter, np.int32(pos[0]), np.int32(pos[1]))) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

        cv2.imshow('frame',frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break       

    output.close()

def detect_img():
    print("Running")
    
    #Load the image
    image = cv2.imread('example.png')
    
    #Detect the lines
    lines = cv2.HoughLines(image, 1, math.pi / 180, 150, None, 0, 0)
    
    #Get the two biggest vertical lines
    auxiliar_lines = get_biggest_lines(get_vertical_lines(lines))
    
    #Calculate the vanishing point
    x,y = get_vanishing_point(auxiliar_lines[0], auxiliar_lines[1])
    
    
    #Detect the players
    players = detect_players()
    
    #Split the players between attackers and defenders
    attackers, defenders = split_attackers_defenders(players)
    
    #Recognice which of the attackers is the most forward
    attacker = get_player_most_close_to_goal(attackers, (x,y))
    
    #Recognice which of the defenders is the most behindhand
    defender = get_player_most_close_to_goal(defenders, (x,y))
    
    #Take the decision to see if it is offside or not
    decision = is_offside(attacker,defender)
    
    #Draw the lines and show them
    attacker_line, defender_line = get_attacker_defender_lines(attacker, defender)
    
    draw_lines_show_image(auxiliar_lines, attacker_line, defender_line, decision, image)    
    
    return None
    

if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        camshift_tracker(video, "output_camshift.txt")
    elif (question_number == 2):
        particle_tracker(video, "output_particle.txt")
    elif (question_number == 3):
        kalman_tracker(video, "output_kalman.txt")
    elif (question_number == 4):
        of_tracker(video, "output_of.txt")

'''
For Kalman Filter:
# --- init
state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state
# --- tracking
prediction = kalman.predict()
# ...
# obtain measurement
if measurement_valid: # e.g. face found
    # ...
    posterior = kalman.correct(measurement)
# use prediction or posterior as your tracking result
'''

'''
For Particle Filter:
# --- init
# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]
# hist_bp: obtain using cv2.calcBackProject and the HSV histogram
# c,r,w,h: obtain using detect_one_face()
n_particles = 200
init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
f0 = particleevaluator(hist_bp, pos) * np.ones(n_particles) # Evaluate appearance model
weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)
# --- tracking
# Particle motion model: uniform step (TODO: find a better motion model)
np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")
# Clip out-of-bounds particles
particles = particles.clip(np.zeros(2), np.array((im_w,im_h))-1).astype(int)
f = particleevaluator(hist_bp, particles.T) # Evaluate particles
weights = np.float32(f.clip(1))             # Weight ~ histogram response
weights /= np.sum(weights)                  # Normalize w
pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average
if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
    particles = particles[resample(weights),:]  # Resample particles according to weights
# resample() function is provided for you
'''