# Deep Learning - Minor Exam 1 

    Ayush Abrol B20AI052

---

## Connecting to Google Drive

Following code is used to connect to Google Drive. This is done to access the dataset and save the model.

    from google.colab import drive
    drive.mount('/content/drive')

## Setting the runtime type to GPU

Command for viewing GPU details

    !nvidia-smi

Following code is used to run the notebook on GPU. This is done to speed up the training process. Adding the following line of code to the top of the notebook will ensure that the notebook is run on GPU. If GPU is not available, then the notebook will be run on CPU.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

Adding ().to(device) to the end of the line of code will ensure that the tensor is stored on GPU.

## Selecting the dataset from Google Drive from the given path

    import os
    os.chdir('/content/drive/My Drive/Minor_1/dataset')
    !ls

## To run the code

    Press Run All for running the complete jupyter notebook
    Press Shift + Enter for running a single cell

Note: All the details regarding the code are mentioned in the markdowns in the jupyter notebook itself.