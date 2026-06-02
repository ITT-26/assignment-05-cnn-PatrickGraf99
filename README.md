[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/cMaQVOgt)


# Task 1: Hyperparameters

The hyperparameter we are looking at is the image size. The default value used in the course was 64.

Wwe will go as low as 8 and as high as 512 for this test. All code can be followed in the `img_size.ipynb` notebook.
The models are also saved as keras files, should you want to try it yourself and skip the training.

My original theory was that given more inputs the training time should increase and the accuracy as well.

In the evaluation we can see that training time does increase with larger images. This is probably due to the fact that 
more data has to be processed. We can see an image size of 64x64 as an outlier, since training here took longer than for 
128 or 256. At this time I have no explanation for this. I theorize that mabe 64 has just a weird mixture of patterns 
and noise so it takes more time to train but leaves us with good results.

The second thought however can be debunked. We see that our model for size 32 does best in the accuracy department.
In hindsigth it seems logical that adding more data than can not improve anything. Once any algorithm reaches 99%
accuracy one can probably assume that adding more data increases the likeliness of adding noise instead of adding just
the right data to reach 100% accuracy


# Task 2: Gathering a dataset

To use the annotater simply make any gesture you want to capture and press f to freeze the frame. Once you have a frozen
frame you can use the mouse to select hands, that means click once to select the top left corner of the bounding box and
click again to select the bottom right corner of the bounding box. After that enter a label in the console and press 
Enter. Repeat this step for every hand in the picture. Once you are done press s. The image will then be saved and the 
bboxes and labels will be added to annot-name.json