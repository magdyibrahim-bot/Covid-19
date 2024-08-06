import numpy as np
# to read excel sheet(.csv)
import pandas as pd 
#scikit-learn's train_test_splitز
from sklearn.model_selection import train_test_split
#tandardScaler for data preprocessing.
from sklearn.preprocessing import StandardScaler

# read the dataset.
# The cancer dataset is loaded from a CSV file into a Pandas DataFrame.
df = pd.read_csv('Cancer_Data.csv')
pd.set_option('display.max_columns', None) # Sets the display option to show all columns of the DataFrame.
pd.set_option('display.max_rows', None)  # Sets the display option to show all rows of the DataFrame.
pd.set_option('display.max_colwidth', None) # Sets the display option to show full column width.

# The target variable 'diagnosis' is converted to numerical labels (0 and 1).
# 0 => hasn't Cancer and 1 => has Cancer
mapForDiagnosis = ['B', 'M']
#Converts the diagnosis column to numerical labels (0 for benign, 1 for malignant) using a mapping.
for row in df.itertuples():
    if((df.at[row.Index, 'diagnosis'] not in mapForDiagnosis)):
        mapForDiagnosis.append(df.at[row.Index, 'diagnosis'])
        df.at[row.Index, 'diagnosis'] = mapForDiagnosis.index(df.at[row.Index, 'diagnosis'])
    else:
        df.at[row.Index, 'diagnosis'] = mapForDiagnosis.index(df.at[row.Index, 'diagnosis'])

# Splitting data
# The features and target are split into training and testing sets using train_test_split.
# Splits the dataset into features X (input variables) and target Y (output variable).
# Features that into the .csv File
X = df[['radius_mean',	'texture_mean',	'perimeter_mean',	'area_mean',	
        'smoothness_mean',	'compactness_mean',	'concavity_mean',	'concave points_mean',	
        'symmetry_mean',	'fractal_dimension_mean',	'radius_se',	'texture_se',	
        'perimeter_se',	'area_se',	'smoothness_se',	'compactness_se',	'concavity_se',	
        'concave points_se',	'symmetry_se',	'fractal_dimension_se',	'radius_worst',	'texture_worst',	
        'perimeter_worst',	'area_worst',	'smoothness_worst',	'compactness_worst',	'concavity_worst',	
        'concave points_worst',	'symmetry_worst',	'fractal_dimension_worst']]
Y = df[['diagnosis']]# Diagnosis in (.csv) file 

# 30% test and 70% training
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=5)
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# Regularization
# The features are standardized using StandardScaler.
# To pre-Processing Data
stdObject = StandardScaler()
x_train = stdObject.fit_transform(x_train)
x_test = stdObject.transform(x_test)

########################################################################################################

# (x_train.shape,y_train.shape)used to check the dimensions of the training data before further processing or
# to ensure that the data is correctly loaded and formatted.
# Initializes weights W and bias b with zeros.
W = np.zeros((x_train.shape[1], 1))
b = 0
x_train.shape
y_train.shape
# computes the linear transformation of the input features X using the weights W and bias b
# X: This parameter represents the input features.
# W: This parameter represents the weights of the model.
# b: This parameter represents the bias term of the model.
# The function computes the dot product of the input features X and the weight matrix W,
# adds the bias term b,and returns the resulting value.
def compute_f_wb(X, W, b): # y = xw + b
    value = np.dot(X, W) + b # (dot) is a NumPy function used to compute the dot product of two arrays.
    return value

""" 
=> The sigmoid function is commonly used in logistic regression
   to squash the output of linear functions into the range [0, 1],
   which is interpreted as the probability of belonging 
   to a certain class in binary classification problems.

=> if less than 0.5 it belongs to one class
   else it belongs to onther class.
 """
def compute_sigmoid(z):
    res = 1/(1+np.exp(-z.astype(int))) # exp(x) = e^x
    # astype(int) => This converts the elements of the array z to integers.
    return res

""" 
=>  calculates the logistic regression cost function,
    also known as the cross-entropy loss function,
    for a given set of parameters W, b bias, input features XX,
    and target labels Y
 """
def compute_cost(W, b, X, Y):
    m = X.shape[0]
    err = 0
    for i in range(m):
        # get the sigmoid for each sample in x
        z = compute_sigmoid(compute_f_wb(X[i, :], W, b))
        if Y[i] == 0:
            curr_error = np.log(1-z+0.0000000001)
        else:
            curr_error = np.log(z+0.0000000001)
        err+= curr_error
    err = err/ (-m)
    return err
compute_cost(W, b, x_train, y_train)

def compute_dj_dw(X, Y, W, b):
    m = X.shape[0]
    err = 0
    for i in range(m):
        z = compute_sigmoid(compute_f_wb(X[i, :], W, b))
        current_err = (z-Y[i])*X[i]
        err+=current_err
    err/=m
    return err.reshape(err.shape[0], 1)

compute_dj_dw(x_train, y_train, W, b).shape
compute_dj_dw(x_train, y_train, W, b).shape

def compute_dj_db(X, Y, W, b):
    m = X.shape[0]
    err = 0
    for i in range(m):
        z = compute_sigmoid(compute_f_wb(X[i, :], W, b))
        current_err = (z-Y[i])
        err+=current_err
    err/=m
    return err

compute_dj_db(x_train, y_train, W, b).shape
compute_dj_db(x_train, y_train, W, b)

def GradientDescent(X, Y, W, b, learning_rate, iters):
    j_hist = []
    j_hist.append(compute_cost(W, b, X, Y))
    """ print(f'J_wb: {j_hist[0]}') """
    for i in range(iters):
        dj_db = compute_dj_db(X, Y, W, b)
        dj_dw = compute_dj_dw(X, Y, W, b)
        W = W - dj_dw*learning_rate
        b = b - dj_db[0]*learning_rate
        j = compute_cost(W, b, X, Y)
        #print(f'J_wb: {j}')
        j_hist.append(j)
    return W, b, j_hist

current_W, current_b, j_hist = GradientDescent(x_train, y_train, W, b, 0.01, 100)
""" for i in j_hist:
    print(i) """


def predict(X, W, b):
    f_wb = compute_f_wb(X, W, b)
    results = compute_sigmoid(f_wb)
    return results

# yhat = predict(x_test, W, b)
len(j_hist)
W.shape
""" for i in current_W:
    print(i) """

current_b
yhat = predict(x_test, current_W, current_b)
yhat.shape

for i in range(yhat.shape[0]):
    if yhat[i] < 0.5:
        yhat[i] = 0
    else:
        yhat[i] = 1

accuracy = 0
for i in range(yhat.shape[0]):
    if yhat[i] == y_test[i]:
        accuracy+=1
accuracy /=yhat.shape[0] 

print(f'The accuracy is: {accuracy*100}\nNumber of correctly classified labels: {accuracy*yhat.shape[0]}\nThe number of misclassified labels: {y_test.shape[0]- accuracy*yhat.shape[0]}')