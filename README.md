# Capstone Project (Fall 2020)
This project is the work of Camille Taltas, Tinantin Nikvashvili, and Francesca Guiso, under the supervision of Professor Kyunghyun Cho and Doctor Eric Oermann. The goal of this project was to simulate an MOS6502 processor run on a Donkey Kong game using a neural network which inferred its predicitions from the actual structure of the chip and applying this neural network to human intracranial recordings in order to better understand the functioning of the brain. The report on this project can be found in Capstone_Final.pdf. The PCA.py, hist.py, files are code to run our data exploration. The emu_data.py, sim6502_data.py, simTIA_data.py, and final_data.py, are the scripts which parse, clean, and aggregate the simulator data, which was provided by Eric Jonas' simulator linked in the report. The modeling.py file is the code for our model trained on the training set, optimized on the validation set, and run on the test set as well as an iterative loop which reports how well our model performs over several times steps of the chip. The generate_ims.py and nn_generate_ims.py are scripts which use the predictions from our neural network to evaluate if our model is capable of recreating the images of the game with the appropriate datapoints. Lastly, all the shell scripts were used to run our code with GPUs on the NYU Prince cluster.
