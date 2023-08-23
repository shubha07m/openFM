import pandas as pd
import part1_basefunctions

# Loading the raw alarm data for training
data = pd.read_csv('All_alarm_stage1.csv')

# # Checking cross validation performance
part1_basefunctions.modelselect(data)
print('\n')

# Perform the training, saving the trained model and checking test data performance
modelname = 'falsealarmdetection.sav'
part1_basefunctions.modeltraining(modelname, data)
#
# # Checking the ROC performance
part1_basefunctions.ROCcurve(data)
#
# # Loading out of sample data
oos_data = pd.read_csv('All_alarm_validation.csv')
print('\n')
#
# # Checking the performance on fresh out of sample data

part1_basefunctions.out_ofsample_perf(modelname, oos_data)
