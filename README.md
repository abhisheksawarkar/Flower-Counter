# Flower-Counter

Counts number of flowers in an image, given the template of the particular flower.
Uses Histogram Backprojection algorithm to do template matching
Used OpenCV for IO and Morphological operations. However, the main algoritm is coded without using inbuilt functions.

Procedure -

1. mkdir build && cd build
2. cmake ..
3. make
4. ./main template.jpg source.jpg

template.jpg and source.jpg can be selected from Test cases folder.
