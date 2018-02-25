"""
This python program detects sidewalks and gives real time navigation to a blind man using the camera. It uses a python module called OpenCV for edge detection and numpy for Machine Learning. Points are taken from a certain region of interest which the area below the webcam. Linear Regression gives us line from two clusters of points(from both sides of the sidewalk. Nonlinear regression also gave us two curves from these two clusters. The point of intersection determines the descision of going left,right or straight, which is stored in a text file called write.txt The second file say.py takes the data from the text file and outputs it by voice using a module called pysttx3

The video provided can be used to display this, by default it uses the computer's webcam. To do this change
cap = cv2.VideoCapture(0)

to
cap = cv2.VideoCapture("challenge.mp4")

To run the program, two python files main.py and say.py should be run simultaneously.
Required Modules

These can be installed by running pip install modulename

    Opencv
    Numpy
    Scipy
    pyttsx3


Math and Time modules are used, but inbuilt.
Displayed Lines

The picture below shows the algorithm working. The dark blue lines is the output that the edge detection algorithm gave. Quadratic regression gives the light blue lines and linear regression gives the black lines.

Achievements

This program won 3rd place for Tech Cares at CruzHacks 2018.
https://devpost.com/software/navigation-for-the-blind-kdyufq
"""

import cv2
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
import math
from scipy.optimize import curve_fit


def intersection(line,line1):
    #if(line[0]-line1[0]) == 0:
    #    return None
    x = (line1[1]-line[1])/(line[0]-line1[0])
    if(x <= 800 and x >= 0) and (((line[0]*x) + line[1]) >= 0) and ((line[0]*x) + line[1] <= 600):
        return [x, (line[0]*x) + line[1]]
def turnPercent(m):
    alpha = math.atan(m)
    alpha = alpha*180 / math.pi
    if (int(alpha) > 0):
        return int(90/alpha *100), "right"
    else:
        alpha = 180-alpha
        return int(90/alpha *100), "left"
        

def getSlope(x1, y1, x2, y2):
    m = (y2-y1) / (x2-x2)
    return m


def polyReg(xcors, ycors):
    def func(x, a, b, c):
        return (a*(x**2)) + (b*x) + c
    time = np.array(xcors)
    avg = np.array(ycors)
    initialGuess = [5, 5, -.01]
    guessedFactors = [func(x, *initialGuess) for x in time]
    popt, pcov = curve_fit(func, time, avg, initialGuess)
    cont = np.linspace(min(time), max(time), 50)
    fittedData = [func(x, *popt) for x in cont]
##    fig1 = plt.figure(1)
##    ax = fig1.add_subplot(1, 1, 1)
##    ax.plot(time, avg, linestyle='', marker='o', color='r', label="data")
##    ax.plot(cont, fittedData, linestyle='-', color='g', label="model")
    xcors = []
    ycors = []
    for count, i in enumerate(cont):
       xcors.append(i)
       ycors.append(fittedData[count])
       
##    ax.legend(loc=0, title="legend", fontsize=12)
##    ax.set_ylabel("ycors")
##    ax.set_xlabel("xcors")
##    ax.grid()
##    ax.set_title("Edges detected")
##    plt.show()
    return popt, xcors, ycors

def linearRegression(X, Y):
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    m = len(X)
    numer = 0
    denom = 0
    for i in range(m):
        numer += (X[i] - mean_x) * (Y[i] - mean_y)
        denom += (X[i] - mean_x) ** 2
    b1 = numer / denom
    b0 = mean_y - (b1 * mean_x)

    max_x = np.max(X) + 5
    min_x = np.min(X) - 5
    x = np.linspace(min_x, max_x, 1000)
    y = b0 + b1 * x

    return b1, b0

def average(diction):
    xcors1 = 0
    ycors1 = 0
    xcors2 = 0
    ycors2 = 0
    count = 0
    for data in diction:
        xcors1 = xcors1 + data[2][0]
        ycors1 = ycors1 + data[2][1]
        xcors2 = xcors2 + data[2][2]
        ycors2 = ycors2 + data[2][3]
        count = count + 1
    xcors1 = xcors1/count
    ycors1 = ycors1/count
    xcors2 = xcors2/count
    ycors2 = ycors2/count

    return (int(xcors1), int(ycors1), int(xcors2), int(ycors2))
        

def averageLanes(lines):
    try:    
        ycor = []

        for i in lines:
            for x in i:
                ycor.append(x[1])
                ycor.append(x[3])

        minY = min(ycor)
        maxY = 600
        linesDict = {}
        finalLines = {}
        lineCount = {}
        for count, i in enumerate(lines):
            for x in i:

                xcors = (x[0], x[2])
                ycors = (x[1], x[3])
                
                A = vstack([xcors,ones(len(xcors))]).T
                m, b = lstsq(A, ycors)[0]

                x1 = (minY-b) / m
                x2 = (maxY-b) / m
               
                linesDict[count] = [m,b,[int(x1), minY, int(x2), maxY]]
                



        status = False
        for i in linesDict:
            finalLinesCopy = finalLines.copy()

            m = linesDict[i][0]
            b = linesDict[i][1]

            line = linesDict[i][2]
            
            if len(finalLines) == 0:
                finalLines[m] = [[m,b,line]]
            else:
                status = False
                
                for x in finalLinesCopy:
                    if not status:
                        if abs(x*1.2) > abs(m) > abs(x*0.8):
                            if abs(finalLinesCopy[x][0][1]*1.2) > abs(b) > abs(finalLinesCopy[x][0][1]*0.8):
                                finalLines[x].append([m,b,line])
                                status = True
                                break
                                
                        else:
                            finalLines[m] = [[m,b,line]]


        for i in finalLines:
            lineCount[i] = len(finalLines[i])

        extremes = sorted(lineCount.items(), key=lambda item: item[1])[::-1][:2]
        lane1 = extremes[0][0]
        lane2 = extremes[1][0]
        
        l1x1, l1y1, l1x2, l1y2 = average(finalLines[lane1])
        l2x1, l2y1, l2x2, l2y2 = average(finalLines[lane2])

        allxcors = [[l1x1, l1x2], [l2x1, l2x2]]
        allycors = [[l1y1, l1y2], [l2y1, l2y2]]

        return allxcors, allycors
        
        

        
    except Exception as e:
        pass
        
def roi(img, vert):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vert, 255)
    return cv2.bitwise_and(img, mask)

def filterWhite(img):
##    lower = np.uint8([200, 200, 200])
##    upper = np.uint8([255, 255, 255])
##    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
##    white_mask = cv2.inRange(img, lower, upper)
##    lower = np.uint8([255,150,0])
##    upper = np.uint8([255, 255, 255])
##    yellow_mask = cv2.inRange(img, lower, upper)
##    mask = cv2.bitwise_or(white_mask, yellow_mask)
##    res = cv2.bitwise_and(img, img, mask = mask)
##    cv2.imshow("res", res)


    lower_white = np.uint8([200, 200, 200])
    upper_white = np.uint8([255, 255, 255])
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img, lower_white, upper_white)
    res = cv2.bitwise_and(img, img, mask = mask)
    cv2.imshow("res", res)


    
    return res    

def getPoints(lines):
    xcors = []
    ycors = []
    for i in lines:
        xcors.append(i[0][0])
        ycors.append(i[0][1])
        xcors.append(i[0][2])
        ycors.append(i[0][3])
    return xcors, ycors

def edgeDetect(img):
     edges = cv2.Canny(img, 250, 300)
     return cv2.GaussianBlur(edges, (3,3), 0)
    
    
def run(screen):
    fin = screen
    #fin = filterWhite(screen)
    vert = np.array([[100,550],[375,350],[450,350],[800,550]], np.int32)
       
    fin = edgeDetect(fin)
    fin = roi(fin, [vert])
    
    line = cv2.HoughLinesP(fin, 2, np.pi/180, 20, 7, 7)
    if not(line is None):
        for i in line:
            cv2.line(screen, (i[0][0], i[0][1]) , (i[0][2], i[0][3]), (255,0,0), 10)
##        cv2.imshow("edgedetection", fin)
    
    l1dataset = []
    l2dataset = []
    try:
        straightxcors, straightycors = averageLanes(line)
        xcors, ycors = getPoints(line)
        
        l1dataset.append(straightxcors[0])
        l1dataset.append(straightycors[0])
        l2dataset.append(straightxcors[1])
        l2dataset.append(straightxcors[1])
        allstraightxcors = straightxcors[0] + straightxcors[1] 
        allstraightycors = straightycors[0] + straightycors[1]

        l1m ,l1b = linearRegression(l1dataset[0], l1dataset[1])
        l2m ,l2b = linearRegression(l2dataset[0], l2dataset[1])
    
        allm, allb = linearRegression(allstraightxcors, allstraightycors)
        allxcor1 = int((allm * 350) + allb)        
        allxcor2 = int(allb)
        
        filterl1x = []
        filterl1y = []
        filterl2x = []
        filterl2y = []

#       cv2.line(screen, (int(allxcor1),600), ( int(allxcor2),400), (255,255,255), 10)
#           for count, i in enumerate(ycors):
#               if (i*allm + allb < xcors[count]):
#                   filterl2x.append(xcors[count])
#                   filterl2y.append(i)
#               else:
#                   filterl1x.append(xcors[count])
#                   filterl1y.append(i)
        

        for count, i in enumerate(ycors):
            if (i*l2m + l2b < xcors[count]):
                filterl2x.append(xcors[count])
                filterl2y.append(i)
            else:
                filterl1x.append(xcors[count])
                filterl1y.append(i)
        
        l1inx1 = int((600 - l1b) / l1m)
        l1inx2 = int((350-l1b) / l1m)
        
        l2inx1 = int((600-l2b) / l2m)
        l2inx2 = int((350-l2b) / l2m)
        
        cv2.line(screen, (int(l1inx1),600), ( int(l1inx2),350), (0,0,0), 10)
        cv2.line(screen, (int(l2inx1),600), ( int(l2inx2),350), (0,0,0), 10)

        #cv2.line(screen, (allxcor1, 600), (allxcor2,350), (255,0,0), 10)
        #turning = ""

        results = intersection([l1m,l1b], [l2m, l2b])
        if (results is not None):
            if (results[0] > 400):
                with open("write.txt","w") as f:
                    f.write("Turn Left")
            else:
                with open("write.txt","w") as f:
                    f.write("Turn Right")
        #else:
        #    with open("write.txt","w") as f:
        #        f.write("Go straight")
            
        try:
            equ1, polyx1, polyy1 = polyReg(filterl2x, filterl2y)
            #print(str(equ1[0]) + "x^2 + " + str(equ1[1]) + "x + " + str(equ1[2]))
            #intersect1 = findIntersection(l2m, l2b, equ1[0], equ1[1], equ1[2])
            #print(intersect1)
            for i in range(len(polyx1)):
                if i == 0:
                    pass
                else:
                    cv2.line(screen, (int(polyx1[i]), int(polyy1[i])), (int(polyx1[i-1]), int(polyy1[i-1])), (255,255,0), 10)
        except Exception as e:
            print(e)
        try:
            equ2, polyx2, polyy2 = polyReg(filterl1x, filterl1y)
            
            for i in range(len(polyx2)):
                if i == 0:
                    pass
                else:
                    cv2.line(screen, (int(polyx2[i]), int(polyy2[i])), (int(polyx2[i-1]), int(polyy2[i-1])), (255,255,0), 10)
        except:
            pass
    except Exception as e:
        pass
    
    return screen


cap = cv2.VideoCapture("challenge.mp4")

while True:
    try:
        screen = cap.read()[1]
        screen = cv2.resize(screen, (800,600))
        cv2.imshow("Test", run(screen))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    except cv2.error:
        exit()

