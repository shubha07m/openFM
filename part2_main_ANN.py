import pandas as pd
import part2_basefunctions_ANN
from datetime import datetime

# Loading the raw alarm data for training
data = pd.read_csv('All_alarm_stage2.csv')

# Checking cross validation performance
part2_basefunctions_ANN.deepmodelselect(data)
print('\n')
t1 = datetime.now()
# Perform the training, saving the trained model and checking test data performance
part2_basefunctions_ANN.modeltraining(data)
t2 = datetime.now()
print(t2-t1)
# Reloading the trained model saved locally
model = 'deeptesting.json'
weight = 'deeptesting.h5'
# Loading out of sample data
oos_data = pd.read_csv('part2_alarm_validation.csv')
print('\n')

# Checking the performance on fresh out of sample data
part2_basefunctions_ANN.out_ofsample_perf(oos_data, model, weight)
