## Myops Challenge 2020 

This project focusses on the the MyOPS2020 challenge. The focus was to create and train a model that was able to recognize edema and scarring on the left ventricle of heart on cardiac MRI's. 

## The model in a Nutshell 

You can see the main architecture of the model below.
![image](https://github.com/jonathan-meulens/Data/assets/120602760/bea55143-2c23-48a9-af57-28277caa55bc)

Figure_1 : Its quite similar to U-NET; However , it passes the skip connections that normally would go to the decoder side,
to the BiFPN (Bidirectional Feature Pyramid Network) first, before going to the decoder side.

The main idea , was to create brand new adaptation of the U-NET , to enhance prediction accuracy by increasing the resolution of the skip connections overall to, re-capture some additional "global" information at the lower layers of the encoder. 

You can see main principle of BiFPN below:
![image](https://github.com/jonathan-meulens/Data/assets/120602760/e86c93db-2813-4af2-8ab7-8a7d4cdcb9d5)

Figure_2 : The depiction of one bifpn-block performing fusion in efficientdet. It combines input features from different levels (p1,…….pn) and finds proper weights to merge higher-resolution inputs with lower-resolution inputs. BiFPN efficiently fuses features at different resolutions to enhance the overall performance of the network.

## Usage 

First off , to run this code , you will need to download the dataset available at https://zmiclab.github.io/zxh/0/myops20/

If you are not familiar with machine learning pipelines, or just want to get the architecture up and running as soon as possible. What you are going to need to CONSIDER is the following: 

1. Download all the .py files ofcourse.
   
2. Check the main.py and main_2.py file. The main.py file will contain all the functions needed to train the plain U-NET. This means that you will NOT be using the red (BiFPN block) in this training section. This is beacause you are only trying to isolate , or train the model to recognize the area of interest; which in this case, is the LEFT VENTRICLE of the heart (in the BssFP/ C1-images). When you train the model to recognize the area of interest, then, use main_2.py file.
    
3. Now that your model knows where the left ventricle is , you can use the architecture above (BiFPN) to enhance the "global" resolution of the images no matter the layer they are in. This is the main_2.py file. Run it.

Additional Note: Make sure to transfer the weights you get from the first model (main.py) to the next model (main_2.py). You can do this by using the LOAD_CHECKPOINT(model_1) function, to load the weights of first (UNET) model (main.py) to the second (BiFPN) model (main_2.py).





