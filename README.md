# Navigation For the Blind
This python program detects sidewalks and gives real time navigation to a blind man using the camera. It uses a python module called OpenCV for edge detection and numpy for Machine Learning. Points are taken from a certain region of interest which the area below the webcam. Linear Regression gives us line from two clusters of points(from both sides of the sidewalk. Nonlinear regression also gave us two curves from these two clusters. The point of intersection determines the descision of going left,right or straight.
<br /><br />
The video provided can be used to display this, by default it uses the computer's webcam. To do this change
<br />
`cap = cv2.VideoCapture(0)`
<br /><br />
to
<br />
`cap = cv2.VideoCapture("challenge.mp4")`
<br /><br />
![alt text](https://github.com/vivek3141/NavigationForBlind/blob/master/Documentation/road.PNG)
