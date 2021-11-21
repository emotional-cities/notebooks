# notebooks

Data analysis examples for the eMOTIONAL Cities project. Data used in the notebooks has been made publicly available in Amazon Simple Storage Service (S3) buckets.

## Compatible Environments

[Visual Studio Code](https://code.visualstudio.com/): All notebooks have been tested in Visual Studio Code on a Windows platform. Tests in other platforms and environments are forthcoming and will be added here.

## How to build

The current notebook requires Python 3.9+ to run successfully. The file `requirements.txt` in the `python` folder contains the list of minimal package dependencies required.

### Windows (self-contained)

For Windows environments, a self-contained `setup.ps1` is included in the `python` folder which can be used to download a self-contained WinPython distribution which will automatically configure all required dependencies.
