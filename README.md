# image_cnn


run "pip3 install -r requirements.txt" command to install all the dependency.
For better accuracy of your model you shold keep different samples inside your training and test data_set.
Change the dense of the last layer of the nn model according to your sample size.
I had took 5 type of sample to train and feed the network that's why my dense of the last layer is 5.
Create a directory with the name of dataset and inside that create 2 new diretory with the name training_set and test_set, inside that 2 directory place your sample directories.
