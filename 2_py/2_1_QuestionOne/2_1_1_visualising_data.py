import pandas as pd

# Reading the data from the .csv file and storing in a panda data frame
dataFilePath = "../../1_data/mushrooms.csv"
data = pd.read_csv(dataFilePath)


print("\n\n\n##### First 5 rows of mushrooms.csv #####\n\n",data.head(),"\n\n\n")
print("##### mushroom.csv information ##### \n\n", data.describe(),"\n\n\n")
print("##### Shape of mushroom.csv #####\n", data.shape,"\n\n\n")
print("##### Any null values? #####\n\n", data.isnull().sum(),"\n\n\n")
