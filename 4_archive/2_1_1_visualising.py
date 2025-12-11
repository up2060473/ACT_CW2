import pandas as pd

# Reading the data from the .csv file and storing in a panda data frame
dataFilePath = "../../1_data/mushrooms.csv"
data = pd.read_csv(dataFilePath)

# Printing summary to terminal
print("\n\n\nVISUALISING DATA.......................")
print("\n\n\n##### First 5 rows of mushrooms.csv #####\n\n",data.head(),"\n\n\n")
print("##### mushroom.csv information ##### \n\n", data.describe(),"\n\n\n")
print("##### Shape of mushroom.csv #####\n", data.shape,"\n\n\n")
print("##### Any null values? #####\n\n", data.isnull().sum(),"\n\n\n")

# Writing summary to text file
with open("./2_1_1_log.txt", "w") as file:
    file.write("VISUALISING DATA..............")
    # Print first 5 rows
    file.write("##### First 5 rows of mushroom.csv ##### \n\n" + data.head().to_string(index=False) + "\n\n\n")
    # Print summary
    file.write("##### mushroom.csv information ##### \n\n"+data.describe().to_string()+"\n\n\n")
    # Print shape 
    file.write("##### Shape of mushroom.csv #####\n\n"+str(data.shape)+"\n\n\n")
    # Print null value summary
    file.write("##### Any null values? #####\n\n"+str(data.isnull().sum())+"\n\n\n")
print("Description written to 2_1_1_log.txt")
