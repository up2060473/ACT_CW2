import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

print("Starting...")
# Reading the data from the .csv file and storing in a panda data frame
print("Reading data...")
dataFilePath = "../../1_data/mushrooms.csv"
data = pd.read_csv(dataFilePath)

# Printing summary to terminal
input1 = input("Print out data summary? Will still write to log file (y/n): ")
if input1 == "y":
    print("\n\n\nVISUALISING DATA.......................")
    print("\n\n\n##### First 5 rows of mushrooms.csv #####\n\n",data.head(),"\n\n\n")
    print("##### mushroom.csv information ##### \n\n", data.describe(),"\n\n\n")
    print("##### Shape of mushroom.csv #####\n", data.shape,"\n\n\n")
    print("##### Any null values? #####\n\n", data.isnull().sum(),"\n\n\n")

# Writing summary to text file
with open("./DecisionTree_log.txt", "w") as file:
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

# Changing values to numbers through label encoding
print("Pre-processing...")
# Initialise a label encoder and empty dictionary to store label encoders
labelEncoder = LabelEncoder()
label_encoders = {}

# Writing to a log file to report on encoding mappings
with open("./DecisionTree_log.txt", "a") as file:
    # Loop over each column
    for column in data.columns:
        # Apply label encoding
        data[column] = labelEncoder.fit_transform(data[column])
        file.write("##### Mapping for "+str(column)+" #####\n")
        # Write mapping to the log file
        for i, class_label in enumerate(labelEncoder.classes_):
            file.write(str(class_label)+" -> "+str(i)+" \n")
        # Store label encoder in a dictionary
        label_encoders[column] = labelEncoder

# Splitting data into features x and labels y
x = data.drop("class", axis=1)
y = data["class"]

# Split data into 60% training and 40% testing 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# Initialise Decision Tree Classifier and train model
print("Training Decision Tree Classifier...\n")
clf = DecisionTreeClassifier(random_state=42)
clf.fit(x_train, y_train)

# Make oredictions on both training and testing data
train_predictions = clf.predict(x_train)
test_predictions = clf.predict(x_test)

#Caclulate accuracy scores for both training an testing
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Training successfull...")
print(">>> Training accuracy: ", train_accuracy)
print(">>> Testing accuracy: ", test_accuracy, "\n")

with open("./DecisionTree_log.txt", "a") as file:
    file.write("\n\n\n ##### Training Accuracy: " + str(train_accuracy) + " #####\n")
    file.write(" ##### Testing Accuracy: " + str(test_accuracy) + " #####\n")

n_nodes = clf.get_depth()
n_leaves = clf.get_n_leaves()
model_params = clf.get_params()
model_params_df_dt_clf = pd.DataFrame([model_params]).T

input2 = input("Print out metrics and model parameters? Will still write to file (y/n):  ")
if input2 == "y":
    print("\n\n\n##### METRICS #####")
    print(">>> number of nodes: ", n_nodes)
    print(">>>number of leaves: ", n_leaves) 
    print("\n\n\n##### MODEL PARAMETERS: #####")
    print(model_params_df_dt_clf)

with open("./DecisionTree_log.txt", "a") as file:
    file.write("\n\n\n##### METRICS #####")
    file.write("\n>>>number of nodes: "+str(n_nodes))
    file.write("\n>>> number of leaves: "+str(n_leaves))
    file.write("\n\n\n#### MODEL PARAMETERS #####")
    file.write("\n"+str(model_params_df_dt_clf))    
        