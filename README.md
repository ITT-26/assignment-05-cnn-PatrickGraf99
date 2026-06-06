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

**I have not tested this script thoroughly yet, but I have deep trust that it does what I expect it to**

## Example
Start the script and make any gesture, press f to freeze the frame

on the frame click once to select the top left point of the bounding box around your hand. Click again, selecting the 
bottom right corner this time

When prompted by the console, type the label you want to give (e.g. like) and then Enter

Do this for every hand and press s. This will save the image as 'uuid.jpg' and add the annotations to annot-name.json

After pressing s the program resets and you can start over

Press q to quit


# Task 3: Gesture controlled camera app

The camera app should be launched using the `camera_launcher.py` and can take 3 named args. -o for the output dir of 
any saved picture (output directory will be same as camera_app.py by default), -m for the model the cnn should use
(The selection is between all those I trained in task 1, the default is '64' since it performed best) and -c for the 
camera id. I recommend leaving them all blank, no arg is required

Launch the script with e.g. `python path/to/camera_launcher.py -m 64` 

Disclaimer: Initializing may take some time, especially on the first run (at least that was the case for me). Don't 
worry, the app works. It is vital you have the .keras files in 01-hyperparameters and the hand_landmarker.task file
in 03-camera-app. All files are included in the git repo.

Control schemes are offered via gestures or keyboard input. By default, the gesture recognition is **turned off** and
has to be enabled by pressing `p` after starting the app. Here is a table displaying the control schemes

| Action                         | keyboard | Gesture   |
|--------------------------------|----------|-----------|
| Toggle gesture controls on/off | `p`      | None      |
| Quit application               | `q`      | None      |
| Toggle Grayscale filter        | `g`      | `like`    |
| Toggle blur                    | `b`      | `dislike` |
| Toggle canny/edge filter       | `c`      | `peace`   |
| Start selfie timer             | `t`      | `fist`    |

The app has an automatic detection pause for 3 seconds after a gesture is detected. The keyboard controls remain 
unaffected by this. 

Applying one filter does not reset other filter, multiple filters can be active at the same time.

When the selfie timer is started, the controls (apart from quitting the application) will be blocked. It is not a bug if
they do not work but intended that way.

The selfie timer is set to 3 seconds (and can right now only be changed by modifying the value in the code)

All selfies taken will have their date and time as a filename so you can easily find them
