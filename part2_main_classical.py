from datetime import datetime
import pandas as pd
import part2_basefunctions_classical

# Loading the raw alarm data for training
data = pd.read_csv('All_alarm_stage2.csv')




# Checking cross validation performance
part2_basefunctions_classical.modelselect(data)
print('\n')

t1 = datetime.now()
modelname = 'part2_classicalmodel.sav'

# Perform the training, saving the trained model and checking test data performance
part2_basefunctions_classical.modeltraining(data, modelname)

t2 = datetime.now()

print(t2-t1)

# Loading out of sample data
oos_data = pd.read_csv('part2_alarm_validation.csv')
print('\n')
# Checking the performance on fresh out of sample data
part2_basefunctions_classical.out_ofsample_perf(modelname, oos_data)