## Task 2, study the KNN



- Define different K



from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions

k_range = [1, 2, 3, 5, 10, 20]
# Initialize array to store the error rate for different dimensions, N
emp_error_2n_2c = []
emp_error_3n_2c = []
emp_error_4n_2c = []
emp_error_5n_2c = []



### 2.1: C=2, 80 samples for each class (40 for training, 40 for testing): Evaluate and plot error rate (@ testing data) for varying K   (consider at least 4 different dimension : N )



#### 2 dimensional for 2 different classes



# Covariance matrices
cov_mats = {}
for i in range(1, 3):
    cov_mats[i] = i * eye(2)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 3), [[0, 0], [3, 0], [4, 5]]):
    mu_vecs[i] = array(j).reshape(2, 1)

# Generating the random samples
all_samples = {}
for i in range(1, 3):
    # generating 80x3 dimensional arrays with random Gaussian-distributed samples
    class_samples = random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], 80)
    # adding class label to 3rd column
    class_samples = append(class_samples, zeros((80, 1))+i, axis=1)
    all_samples[i] = class_samples

# Dividing the samples into training and test datasets
train_set = append(all_samples[1][0:40], all_samples[2][0:40], axis=0)
test_set = append(all_samples[1][40:80], all_samples[2][40:80], axis=0)

assert(train_set.shape == (80, 3))
assert(test_set.shape == (80, 3))

X_train = train_set[:, :-1]
X_test = test_set[:, :-1]
y_train = train_set[:, -1].astype(integer)
y_test = test_set[:, -1].astype(integer)


for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    num_correct_predictions = (y_pred == y_test).sum()
    accuracy = (num_correct_predictions / y_test.shape[0]) * 100
    print('##############################################################################')
    print(f"when k is {k}  !")
    print('Test set accuracy: %.2f%%' % accuracy)
    
    
    plot_decision_regions(X_test, y_test, knn_model)
    plt.legend(loc='upper left')
    plt.title(f'k is {k}: Decision Boundary for Test set')
    plt.show()



#### 3 dimensional for 2 different classes



# Covariance matrices
cov_mats = {}
for i in range(1,3):
    cov_mats[i] = i * eye(3)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 3), [[0, 0, 0], [3, 0, 3], [4, 5, 4]]):
    mu_vecs[i] = array(j).reshape(3, 1)

# Generating the random samples
all_samples = {}
for i in range(1, 3):
    # generating 80x4 dimensional arrays with random Gaussian-distributed samples
    class_samples = random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], 80)
    # adding class label to 3rd column
    class_samples = append(class_samples, zeros((80, 1))+i, axis=1)
    all_samples[i] = class_samples

# Dividing the samples into training and test datasets
train_set = append(all_samples[1][0:40], all_samples[2][0:40], axis=0)

test_set = append(all_samples[1][40:80], all_samples[2][40:80], axis=0)

assert(train_set.shape == (80, 4))
assert(test_set.shape == (80, 4))

X_train = train_set[:, :-1]
X_test = test_set[:, :-1]
y_train = train_set[:, -1]
y_test = test_set[:, -1].astype(integer)

for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    num_correct_predictions = (y_pred == y_test).sum()
    accuracy = (num_correct_predictions / y_test.shape[0]) * 100
    print('##############################################################################')
    print(f"when k is {k}  !")
    print('Test set accuracy: %.2f%%' % accuracy)
    



#### 4 dimensional for 2 classes



# Covariance matrices
cov_mats = {}
for i in range(1, 3):
    cov_mats[i] = i * eye(4)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 3), [[1, 1, 0, 7], [3, 8, 3, 1], [4, 5, 9, 6]]):
    mu_vecs[i] = array(j).reshape(4, 1)

# Generating the random samples
all_samples = {}
for i in range(1, 3):
    # generating 80x4 dimensional arrays with random Gaussian-distributed samples
    class_samples = random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], 80)
    # adding class label to 3rd column
    class_samples = append(class_samples, zeros((80, 1))+i, axis=1)
    all_samples[i] = class_samples

# Dividing the samples into training and test datasets
train_set = append(all_samples[1][0:40], all_samples[2][0:40], axis=0)
test_set = append(all_samples[1][40:80], all_samples[2][40:80], axis=0)

assert(train_set.shape == (80, 5))
assert(test_set.shape == (80, 5))

X_train = train_set[:, :-1]
X_test = test_set[:, :-1]
y_train = train_set[:, -1]
y_test = test_set[:, -1].astype(integer)

for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    num_correct_predictions = (y_pred == y_test).sum()
    accuracy = (num_correct_predictions / y_test.shape[0]) * 100
    print('##############################################################################')
    print(f"when k is {k}  !")
    print('Test set accuracy: %.2f%%' % accuracy)



#### 5 dimensional for 2 classes



# Covariance matrices
cov_mats = {}
for i in range(1, 3):
    cov_mats[i] = i * eye(5)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 3), [[0, 1, 0, 7, 6], [3, 0, 3, 1, 6], [4, 5, 4, 6, 7]]):
    mu_vecs[i] = array(j).reshape(5, 1)

# Generating the random samples
all_samples = {}
for i in range(1, 3):
    # generating 80x4 dimensional arrays with random Gaussian-distributed samples
    class_samples = random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], 80)
    # adding class label to last column
    class_samples = append(class_samples, zeros((80, 1))+i, axis=1)
    all_samples[i] = class_samples

# Dividing the samples into training and test datasets
train_set = append(all_samples[1][0:40], all_samples[2][0:40], axis=0)
test_set = append(all_samples[1][40:80], all_samples[2][40:80], axis=0)

assert(train_set.shape == (80, 6))
assert(test_set.shape == (80, 6))

X_train = train_set[:, :-1]
X_test = test_set[:, :-1]
y_train = train_set[:, -1]
y_test = test_set[:, -1].astype(integer)

for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    num_correct_predictions = (y_pred == y_test).sum()
    accuracy = (num_correct_predictions / y_test.shape[0]) * 100
    print('##############################################################################')
    print(f"when k is {k}  !")
    print('Test set accuracy: %.2f%%' % accuracy)



### 2.2: C=2, 1000 samples for each class (500 for training, 500 for testing): Evaluate and plot error rate (@ testing data) for varying K   (consider at least 4 different dimension : N )



train_size = 1000



#### 2 dimensional for 2 different classes



# Covariance matrices
cov_mats = {}
for i in range(1, 3):
    cov_mats[i] = i * eye(2)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 3), [[0, 0], [3, 0], [4, 5]]):
    mu_vecs[i] = array(j).reshape(2, 1)

# Generating the random samples
all_samples = {}
for i in range(1, 3):
    # generating 80x3 dimensional arrays with random Gaussian-distributed samples
    class_samples = random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], train_size)
    # adding class label to 3rd column
    class_samples = append(class_samples, zeros((train_size, 1))+i, axis=1)
    all_samples[i] = class_samples

# Dividing the samples into training and test datasets, half for train and testing
train_set = append(all_samples[1][0:int((train_size/2))], all_samples[2][0:(int(train_size/2))], axis=0)
test_set = append(all_samples[1][int((train_size/2)):int(train_size)], all_samples[2][int((train_size/2)):
                                                                                         int(train_size)], axis=0)

assert(train_set.shape == (train_size, 3))
assert(test_set.shape == (train_size, 3))

X_train = train_set[:, :-1]
X_test = test_set[:, :-1]
y_train = train_set[:, -1].astype(integer)
y_test = test_set[:, -1].astype(integer)


for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    num_correct_predictions = (y_pred == y_test).sum()
    accuracy = (num_correct_predictions / y_test.shape[0]) * 100
    print('##############################################################################')
    print(f"when k is {k}  !")
    print('Test set accuracy: %.2f%%' % accuracy)
    
    
    plot_decision_regions(X_test, y_test, knn_model)
    plt.legend(loc='upper left')
    plt.title(f'k is {k}: Decision Boundary for Test set')
    plt.show()



#### 3 dimensional for 2 different classes


# Covariance matrices
cov_mats = {}
for i in range(1,3):
    cov_mats[i] = i * eye(3)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 3), [[0, 0, 0], [3, 0, 3], [4, 5, 4]]):
    mu_vecs[i] = array(j).reshape(3, 1)

# Generating the random samples
all_samples = {}
for i in range(1, 3):
    # generating 80x3 dimensional arrays with random Gaussian-distributed samples
    class_samples = random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], train_size)
    # adding class label to 3rd column
    class_samples = append(class_samples, zeros((train_size, 1))+i, axis=1)
    all_samples[i] = class_samples

# Dividing the samples into training and test datasets, half for train and testing
train_set = append(all_samples[1][0:int((train_size/2))], all_samples[2][0:(int(train_size/2))], axis=0)
test_set = append(all_samples[1][int((train_size/2)):int(train_size)], all_samples[2][int((train_size/2)):
                                                                                         int(train_size)], axis=0)

assert(train_set.shape == (train_size, 4))
assert(test_set.shape == (train_size, 4))

X_train = train_set[:, :-1]
X_test = test_set[:, :-1]
y_train = train_set[:, -1]
y_test = test_set[:, -1].astype(integer)

for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    num_correct_predictions = (y_pred == y_test).sum()
    accuracy = (num_correct_predictions / y_test.shape[0]) * 100
    print('##############################################################################')
    print(f"when k is {k}  !")
    print('Test set accuracy: %.2f%%' % accuracy)
    



#### 4 dimensional for 2 classes



# Covariance matrices
cov_mats = {}
for i in range(1, 3):
    cov_mats[i] = i * eye(4)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 3), [[1, 1, 0, 7], [3, 8, 3, 1], [4, 5, 9, 6]]):
    mu_vecs[i] = array(j).reshape(4, 1)

# Generating the random samples
all_samples = {}
for i in range(1, 3):
    # generating 80x3 dimensional arrays with random Gaussian-distributed samples
    class_samples = random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], train_size)
    # adding class label to 3rd column
    class_samples = append(class_samples, zeros((train_size, 1))+i, axis=1)
    all_samples[i] = class_samples

# Dividing the samples into training and test datasets, half for train and testing
train_set = append(all_samples[1][0:int((train_size/2))], all_samples[2][0:(int(train_size/2))], axis=0)
test_set = append(all_samples[1][int((train_size/2)):int(train_size)], all_samples[2][int((train_size/2)):
                                                                                         int(train_size)], axis=0)

assert(train_set.shape == (train_size, 5))
assert(test_set.shape == (train_size, 5))

X_train = train_set[:, :-1]
X_test = test_set[:, :-1]
y_train = train_set[:, -1]
y_test = test_set[:, -1].astype(integer)

for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    num_correct_predictions = (y_pred == y_test).sum()
    accuracy = (num_correct_predictions / y_test.shape[0]) * 100
    print('##############################################################################')
    print(f"when k is {k}  !")
    print('Test set accuracy: %.2f%%' % accuracy)



#### 5 dimensional for 2 classes



# Covariance matrices
cov_mats = {}
for i in range(1, 3):
    cov_mats[i] = i * eye(5)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 3), [[0, 1, 0, 7, 6], [3, 0, 3, 1, 6], [4, 5, 4, 6, 7]]):
    mu_vecs[i] = array(j).reshape(5, 1)

# Generating the random samples
all_samples = {}
for i in range(1, 3):
    # generating 80x3 dimensional arrays with random Gaussian-distributed samples
    class_samples = random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], train_size)
    # adding class label to 3rd column
    class_samples = append(class_samples, zeros((train_size, 1))+i, axis=1)
    all_samples[i] = class_samples

# Dividing the samples into training and test datasets, half for train and testing
train_set = append(all_samples[1][0:int((train_size/2))], all_samples[2][0:(int(train_size/2))], axis=0)
test_set = append(all_samples[1][int((train_size/2)):int(train_size)], all_samples[2][int((train_size/2)):
                                                                                         int(train_size)], axis=0)

assert(train_set.shape == (train_size, 6))
assert(test_set.shape == (train_size, 6))

X_train = train_set[:, :-1]
X_test = test_set[:, :-1]
y_train = train_set[:, -1]
y_test = test_set[:, -1].astype(integer)

for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    num_correct_predictions = (y_pred == y_test).sum()
    accuracy = (num_correct_predictions / y_test.shape[0]) * 100
    print('##############################################################################')
    print(f"when k is {k}  !")
    print('Test set accuracy: %.2f%%' % accuracy)



 
### 2.3 C=5, N = 2, 80 samples for each class (40 for training, 40 for testing): plot error rate  (@ testing data) for varying K to analyze the best K
- Actually, we have done this before.



# Covariance matrices
cov_mats = {}
for i in range(1, 6):
    cov_mats[i] = i * eye(2)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 6), [[0, 0], [3, 0], [4, 5], [3, 6], [7, 3]]):
    mu_vecs[i] = array(j).reshape(2, 1)

# Generating the random samples
all_samples = {}
for i in range(1, 6):
    # generating 80x5 dimensional arrays with random Gaussian-distributed samples
    class_samples = random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], 80)
    # adding class label to 3rd column
    class_samples = append(class_samples, zeros((80, 1))+i, axis=1)
    all_samples[i] = class_samples

# Dividing the samples into training and test datasets
train_set = append(all_samples[1][0:40], all_samples[2][0:40], axis=0)
train_set = append(train_set, all_samples[3][0:40], axis=0)
train_set = append(train_set, all_samples[4][0:40], axis=0)
train_set = append(train_set, all_samples[5][0:40], axis=0)

test_set = append(all_samples[1][40:80], all_samples[2][40:80], axis=0)
test_set = append(test_set, all_samples[3][40:80], axis=0)
test_set = append(test_set, all_samples[4][40:80], axis=0)
test_set = append(test_set, all_samples[5][40:80], axis=0)

# 40 for training and 40 for testing for each class, thus training data and testing data should have
# size of (40 * 5 = 200, 3) shape, last column is the class column
assert(train_set.shape == (200, 3))
assert(test_set.shape == (200, 3))

X_train = train_set[:, :-1]
X_test = test_set[:, :-1]
y_train = train_set[:, -1]
y_test = test_set[:, -1].astype(integer)

for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    num_correct_predictions = (y_pred == y_test).sum()
    accuracy = (num_correct_predictions / y_test.shape[0]) * 100
    print('##############################################################################')
    print(f"when k is {k}  !")
    print('Test set accuracy: %.2f%%' % accuracy)
    



#### 2.4 Analyze how # of training samples (e.g., from 10 to 10k) impact the error rate (@ testing data), with different dimension: N. You can choose other parameters based on your own need.
- Now, we fix K as 5



training_samples = [10, 50, 100, 500, 1000, 5000, 10000]
k = 5
# Initialize array to store the error rate for different dimensions, N
emp_error_2n = []
emp_error_3n = []
emp_error_4n = []



#### 2 dimensional for 2 different classes



# Covariance matrices
cov_mats = {}
for i in range(1, 3):
    cov_mats[i] = i * eye(2)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 3), [[0, 0], [3, 0]]):
    mu_vecs[i] = array(j).reshape(2, 1)

# Generating the random samples
for train_size in training_samples:
    all_samples = {}
    for i in range(1, 3):
        # generating 80x3 dimensional arrays with random Gaussian-distributed samples
        class_samples = random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], train_size)
        # adding class label to 3rd column
        class_samples = append(class_samples, zeros((train_size, 1))+i, axis=1)
        all_samples[i] = class_samples

    # Dividing the samples into training and test datasets, half for train and testing
    train_set = append(all_samples[1][0:int((train_size/2))], all_samples[2][0:(int(train_size/2))], axis=0)
    test_set = append(all_samples[1][int((train_size/2)):int(train_size)], all_samples[2][int((train_size/2)):
                                                                                             int(train_size)], axis=0)

    assert(train_set.shape == (train_size, 3))
    assert(test_set.shape == (train_size, 3))

    X_train = train_set[:, :-1]
    X_test = test_set[:, :-1]
    y_train = train_set[:, -1]
    y_test = test_set[:, -1].astype(integer)
    
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    num_correct_predictions = (y_pred == y_test).sum()
    error = (1-(num_correct_predictions / y_test.shape[0])) * 100

    emp_error_2n.append(error)

#########################################################################
# Plots of the graphs
# First plot, 2 dimensional for 2 classes
fig = plt.figure()

plt.plot(training_samples, emp_error_2n)
plt.title('Error rate vs different training sizes for 2 dimensional feature vectors')
plt.xlabel('Training sizes')
plt.ylabel('Error rate')
plt.grid()
plt.show()




#### 3 dimensional for 2 different classes



# Covariance matrices
cov_mats = {}
for i in range(1,3):
    cov_mats[i] = i * eye(3)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 3), [[0, 0, 0], [3, 0, 3]]):
    mu_vecs[i] = array(j).reshape(3, 1)

for train_size in training_samples:
    all_samples = {}
    for i in range(1, 3):
        # generating 80x3 dimensional arrays with random Gaussian-distributed samples
        class_samples = random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], train_size)
        # adding class label to 3rd column
        class_samples = append(class_samples, zeros((train_size, 1))+i, axis=1)
        all_samples[i] = class_samples

    # Dividing the samples into training and test datasets, half for train and testing
    train_set = append(all_samples[1][0:int((train_size/2))], all_samples[2][0:(int(train_size/2))], axis=0)
    test_set = append(all_samples[1][int((train_size/2)):int(train_size)], all_samples[2][int((train_size/2)):
                                                                                             int(train_size)], axis=0)

    assert(train_set.shape == (train_size, 4))
    assert(test_set.shape == (train_size, 4))

    X_train = train_set[:, :-1]
    X_test = test_set[:, :-1]
    y_train = train_set[:, -1]
    y_test = test_set[:, -1].astype(integer)
    
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    num_correct_predictions = (y_pred == y_test).sum()
    error = (1-(num_correct_predictions / y_test.shape[0])) * 100

    emp_error_3n.append(error)

#########################################################################
# Plots of the graphs
# First plot, 3 dimensional for 2 classes
fig = plt.figure()

plt.plot(training_samples, emp_error_3n)
plt.title('Error rate vs different training sizes for 3 dimensional feature vectors')
plt.xlabel('Training sizes')
plt.ylabel('Error rate')
plt.grid()
plt.show()



#### 4 dimensional for 2 classes

 

# Covariance matrices
cov_mats = {}
for i in range(1, 3):
    cov_mats[i] = i * eye(4)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 3), [[1, 1, 0, 7], [3, 8, 3, 1], [4, 5, 9, 6]]):
    mu_vecs[i] = array(j).reshape(4, 1)

for train_size in training_samples:
    all_samples = {}
    for i in range(1, 3):
        # generating 80x3 dimensional arrays with random Gaussian-distributed samples
        class_samples = random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], train_size)
        # adding class label to 3rd column
        class_samples = append(class_samples, zeros((train_size, 1))+i, axis=1)
        all_samples[i] = class_samples

    # Dividing the samples into training and test datasets, half for train and testing
    train_set = append(all_samples[1][0:int((train_size/2))], all_samples[2][0:(int(train_size/2))], axis=0)
    test_set = append(all_samples[1][int((train_size/2)):int(train_size)], all_samples[2][int((train_size/2)):
                                                                                             int(train_size)], axis=0)

    assert(train_set.shape == (train_size, 5))
    assert(test_set.shape == (train_size, 5))

    X_train = train_set[:, :-1]
    X_test = test_set[:, :-1]
    y_train = train_set[:, -1]
    y_test = test_set[:, -1].astype(integer)
    
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    num_correct_predictions = (y_pred == y_test).sum()
    error = (1-(num_correct_predictions / y_test.shape[0])) * 100

    emp_error_4n.append(error)


#########################################################################
# Plots of the graphs
# First plot, 4 dimensional for 2 classes
fig = plt.figure()

plt.plot(training_samples, emp_error_4n)
plt.title('Error rate vs different training sizes for 2 dimensional feature vectors')
plt.xlabel('Training sizes')
plt.ylabel('Error rate')
plt.grid()
plt.show()




### 2.5 Study the difference of Euclidean distance and Manhattan distance (https://en.wikipedia.org/wiki/Taxicab_geometry), 
you can choose C=2, varying K. You can choose other parameters based on your own need.



- C = 3, N = 5, varying K



#### manhattan distance:



# Covariance matrices
cov_mats = {}
for i in range(1, 6):
    cov_mats[i] = i * eye(3)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 6), [[0, 0, 0], [3, 0, 9], [4, 5, 1], [3, 6, 2], [7, 3, 6]]):
    mu_vecs[i] = array(j).reshape(3, 1)

# Generating the random samples
all_samples = {}
for i in range(1, 6):
    # generating 80x5 dimensional arrays with random Gaussian-distributed samples
    class_samples = random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], 80)
    # adding class label to 3rd column
    class_samples = append(class_samples, zeros((80, 1))+i, axis=1)
    all_samples[i] = class_samples

# Dividing the samples into training and test datasets
train_set = append(all_samples[1][0:40], all_samples[2][0:40], axis=0)
train_set = append(train_set, all_samples[3][0:40], axis=0)
train_set = append(train_set, all_samples[4][0:40], axis=0)
train_set = append(train_set, all_samples[5][0:40], axis=0)

test_set = append(all_samples[1][40:80], all_samples[2][40:80], axis=0)
test_set = append(test_set, all_samples[3][40:80], axis=0)
test_set = append(test_set, all_samples[4][40:80], axis=0)
test_set = append(test_set, all_samples[5][40:80], axis=0)

# 40 for training and 40 for testing for each class, thus training data and testing data should have
# size of (40 * 5 = 200, 3) shape, last column is the class column
assert(train_set.shape == (200, 4))
assert(test_set.shape == (200, 4))

X_train = train_set[:, :-1]
X_test = test_set[:, :-1]
y_train = train_set[:, -1]
y_test = test_set[:, -1].astype(integer)

for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k, metric="manhattan")
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    num_correct_predictions = (y_pred == y_test).sum()
    accuracy = (num_correct_predictions / y_test.shape[0]) * 100
    print('##############################################################################')
    print(f"when k is {k}  !")
    print('Test set accuracy: %.2f%%' % accuracy)



#### Euclidean distance:



# Covariance matrices
cov_mats = {}
for i in range(1, 6):
    cov_mats[i] = i * eye(3)

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 6), [[0, 0, 0], [3, 0, 9], [4, 5, 1], [3, 6, 2], [7, 3, 6]]):
    mu_vecs[i] = array(j).reshape(3, 1)

# Generating the random samples
all_samples = {}
for i in range(1, 6):
    # generating 80x5 dimensional arrays with random Gaussian-distributed samples
    class_samples = random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], 80)
    # adding class label to 3rd column
    class_samples = append(class_samples, zeros((80, 1))+i, axis=1)
    all_samples[i] = class_samples

# Dividing the samples into training and test datasets
train_set = append(all_samples[1][0:40], all_samples[2][0:40], axis=0)
train_set = append(train_set, all_samples[3][0:40], axis=0)
train_set = append(train_set, all_samples[4][0:40], axis=0)
train_set = append(train_set, all_samples[5][0:40], axis=0)

test_set = append(all_samples[1][40:80], all_samples[2][40:80], axis=0)
test_set = append(test_set, all_samples[3][40:80], axis=0)
test_set = append(test_set, all_samples[4][40:80], axis=0)
test_set = append(test_set, all_samples[5][40:80], axis=0)

# 40 for training and 40 for testing for each class, thus training data and testing data should have
# size of (40 * 5 = 200, 3) shape, last column is the class column
assert(train_set.shape == (200, 4))
assert(test_set.shape == (200, 4))

X_train = train_set[:, :-1]
X_test = test_set[:, :-1]
y_train = train_set[:, -1]
y_test = test_set[:, -1].astype(integer)

for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    num_correct_predictions = (y_pred == y_test).sum()
    accuracy = (num_correct_predictions / y_test.shape[0]) * 100
    print('##############################################################################')
    print(f"when k is {k}  !")
    print('Test set accuracy: %.2f%%' % accuracy)
