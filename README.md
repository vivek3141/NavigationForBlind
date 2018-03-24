# Navigation For the Blind
This python program detects sidewalks and gives real time navigation to a blind man using the camera. It uses a python module called OpenCV for edge detection and numpy for Machine Learning. Points are taken from a certain region of interest which the area below the webcam. Linear Regression gives us line from two clusters of points(from both sides of the sidewalk. Nonlinear regression also gave us two curves from these two clusters. The point of intersection determines the descision of going left,right or straight, which is stored in a text file called `write.txt`
The second file `say.py` takes the data from the text file and outputs it by voice using a module called pysttx3
<br /><br />
The video provided can be used to display this, by default it uses the computer's webcam. To do this change
<br />
```python
cap = cv2.VideoCapture(0)
```
<br /><br />
to
<br />
```python
cap = cv2.VideoCapture("challenge.mp4")
```
<br /><br />
To run the program, two python files `main.py` and `say.py` should be run simultaneously.
<br />
## Required Modules
These can be installed by running `pip install modulename`
* Opencv
* Numpy 
* Scipy 
* pyttsx3 
<br />
Math and Time modules are used, but inbuilt.
<br />

## Displayed Lines
The picture below shows the algorithm working. The dark blue lines is the output that the edge detection algorithm gave. Quadratic regression gives the light blue lines and linear regression gives the black lines. 
<br /><br />

### Achievements
This program won 3rd place for Tech Cares at CruzHacks 2018.
<br />
https://devpost.com/software/navigation-for-the-blind-kdyufq


![alt text](https://github.com/vivek3141/NavigationForBlind/blob/master/Documentation/road.PNG)
<br />
<br />

