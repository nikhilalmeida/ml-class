from __future__ import print_function
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import argparse

def readFile(path):
    with open(path, 'rU') as f:
        f.readline()
        data = np.loadtxt(f, delimiter=',')
    return data

def get_rfg_result(training_data, test_data, output):
    rfg_model = RandomForestRegressor()                           # Initializing the model
    print("################# Training the model  ######################")
    rfg_model.fit(training_data[:,0:6], training_data[:,-1])      # Training the model
    print("################# Getting predictions ######################")
    predict_value = rfg_model.predict(test_data[:,1:])            # Getting predictions
    
    with open(output, 'w+') as out:
        print("id", "Demanda_uni_equil", sep=',', file=out)
        for i in range(len(predict_value)):
            print(int(test_data[i,0]), int(round(predict_value[i])), sep=',', file=out)
            
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", "--trainingFile", help="Path to training data")
    parser.add_argument("-test", "--testFile", help="Path to test data")
    parser.add_argument("-o", "--outputFile", help="Path to output file")

    args = parser.parse_args()
    training_data = readFile(args.trainingFile)
    test_data = readFile(args.testFile)
    get_rfg_result(training_data, test_data, args.outputFile)
    

    

    
    
    
