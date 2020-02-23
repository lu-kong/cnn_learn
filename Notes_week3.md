Week 3 Object detection
===

# Object localization

- Classification with localization
- Detection

## Localization?
add bounding box of output layers:
  $$
  b_x, b_y, b_h, b_w
  $$
we give a output form as:
$$
y = [\text{pc},b_x, b_y, b_h, b_w,c_1,c_2,c_3]^T
$$
where pc stand for wether there is some object in the picture  

loss function:
  $$
\begin{aligned}
            \mathcal{L}(\hat{y},y) =\\
      (\hat{y}_1-y_1)^2+(\hat{y}_2-y_2)^2 + ... +(\hat{y}_8-y_8)^2, \quad\text{if}\quad y_1 = 1;\\

      (\hat{y}_1-y_1)^2, \quad\text{if}\quad y_1 = 0;
\end{aligned}
  $$

## Landmark Detection
pick some points important, called `landmark`,  of the object that we want to detect.

## Object Detection
Sliding windows detection : small rectanguler region

- runnning slide windows throughout the picture
- using a larger window to do the same thing
- when there is once the proba of successful detection, we stop because we have already detected the location
- this is in fact very slow in this era

## Convolutional Implementation of Sliding Windows
- Turning FC layer into convolutional layers
  for the last several FC layers in the end of CNN
  - idea: use the number of filter to stand for different neurals of fully connected layers
  - ![](fc2conv.png)
  - 
