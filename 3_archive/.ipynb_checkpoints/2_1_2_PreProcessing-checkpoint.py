import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Reading the data from the .csv file and storing in a panda data frame
print("Reading data...")
dataFilePath = "../../1_data/mushrooms.csv"
data = pd.read_csv(dataFilePath)

# Changing values to numbers through label encoding
print("Pre-processing...")
# Initialise a label encoder and empty dictionary to store label encoders
labelEncoder = LabelEncoder()
label_encoders = {}

# Writing to a log file to report on encoding mappings
with open("./2_1_2_log.txt", "w") as file:
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

print("Training successfull, accuracy results follow...")
print(">>> Training accuracy: ", train_accuracy)
print(">>> Testing accuracy: ", test_accuracy, "\n")

with open("./2_1_2_log.txt", "a") as file:
    file.write("\n\n\n ##### Training Accuracy: " + str(train_accuracy) + " #####\n")
    file.write(" ##### Testing Accuracy: " + str(test_accuracy) + " #####\n")

#from sklearn.tree import plot_tree
#import matplotlib.pyplot as plt

#plt.figure(figsize=(15,10))
#plot_tree(clf, filled=True, feature_names=x.columns, class_names=["edible","poisonous"], rounded=True)
#plt.show()