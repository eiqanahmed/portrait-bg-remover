In order to remove a background from image(s) you would like, put your image(s) in the 'input_images' directory (When you clone the repository onto your own machine, you will see that the 'input_images' folder already has an example image. You can replace that with the images whose background you would like to remove). After doing so, run `remove_bg.py`. Your output shuld be under '/test/final_outputs'. You don't have to install the dataset to test this model (you can directly run remove_bg.py), but if you would like to train it further, consider the following instructions to download the dataset used to train this model.

# Dataset Installation

## People Segmentation Dataset

This project uses the Person Segmentation dataset from Kaggle. Follow these steps to install it:

### Prerequisites
- Kaggle account
- Kaggle API credentials configured

### Installation Steps

1. **Install Kaggle API** (if not already installed): run `pip install kaggle`
2. **Configure Kaggle API credentials**:
   - Go to your Kaggle account settings
   - Create a new API token (download kaggle.json)
   - Place the file in ~/.kaggle/kaggle.json (Linux/Mac) or C:\Users\{username}\.kaggle\kaggle.json (Windows)
3. **Create the data directory in your project root**:
   -  In your project's root directory, run `mkdir data`
4. **Download the dataset** (run the following lines in your terminal, assuming you are in the project's root directory):
   - `cd data`
   - `kaggle datasets download -d nikhilroxtomar/person-segmentation`
5. **Extract the dataset**:
   - `unzip person-segmentation.zip`

The dataset should be available in the 'data' directory for use in your project. 
   

