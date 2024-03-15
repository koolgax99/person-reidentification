# person-reidentification

Make sure [conda](https://www.anaconda.com/distribution/) is installed.

## Creating the dataset from a video

You can crop the videos using the following command, but first will have to install yolo_tracker:
```
    #cd into the yolo_tracking folder in main
    cd yolo_tracking

    # create environment (However, you can create normal virtualenv as well)
    conda create --name yolotracker 
    conda activate yolotracker

    # Install the dependencies
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    
    # Run the follwoing script
    python examples/track.py --tracking-method deepocsort --yolo-model yolov8n.pt --reid-model osnet_ain_x1_0_msmt17 --source input/<video_name> --save --classes 0 --save --save-id-crops --save-mot
```

Once we get the cropped images, you can open the `yolo_tracking/runs/track/<latest-exp>/crops/0` folder and then manually annotate to put them into respective folders. 

Later you can use the `data_helper.py` file to rename the image names. We have to add the id into the name of each image before we can actually start training. 

## Finetuning the model

### Installing torchreid for training
---------------
```

    # clone this repo into the root folder
    git clone https://github.com/KaiyangZhou/deep-person-reid.git

    # create environment
    cd deep-person-reid/
    conda create --name torchreid python=3.7
    conda activate torchreid

    # install dependencies
    # make sure `which python` and `which pip` point to the correct path
    pip install -r requirements.txt

    # install torch and torchvision (select the proper cuda version to suit your machine)
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

    # install torchreid (don't need to re-build it if you modify the source code)
    python setup.py develop
```

### Data

- Create the `train`, `query` and `gallery` folder in the `embodied-learning-data-test` folder in the root repo.
- Move the data into correct folders inside the `train`, `query` and `gallery` folder. 

### Running the script
```
  python main-trainer.py
```

## Testing the trained model on the video

Once we have the trained model. Make sure to follow the step below.
```
    # now change directory to yolo-tracking again
    cd yolo_tracking

    # Run the testing script, but we change the model with our own trained model
    python examples/track.py --tracking-method deepocsort --yolo-model yolov8n.pt --reid-model <custom_model_name.pt> --source input/<video_name> --save --classes 0 --save --save-id-crops --save-mot
```

