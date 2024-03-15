# person-reidentification

## Finetuning the model

### Installing torchreid for training
---------------

Make sure [conda](https://www.anaconda.com/distribution/) is installed.
```

    # cd to your preferred directory and clone this repo
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

