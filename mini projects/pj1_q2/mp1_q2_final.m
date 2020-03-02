clc
clear

% export the data distribution from task 1 into csv and read them
data1 = csvread('data1.csv',1,0);  %data for class 1
data2 = csvread('data2.csv',1,0);  %data for class 2

% initialise array for mean and sigma for both data
mu_1 = []; sigma_1 = []; mu_2 = []; sigma_2 = [];

disp('***Parameter estimation***')
% first part: Number of samples vs parameters estimated
for index = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
    data_indexed_1 = data1(1:index);  %different number of samples for each iteration
    data_indexed_2 = data2(1:index);  %different number of samples for each iteration
    phat1 = mle(data_indexed_1);      %using mle built in function, to generate parameters,
    muhat1 = phat1(1);                %mean and sigma
    sigmahat1 = phat1(2);
    
    phat2 = mle(data_indexed_2);      %repeat same process for data class 2
    muhat2 = phat2(1);
    sigmahat2 = phat2(2); 

    mu_1 = [mu_1 muhat1];             %append mean into mu_1 array
    sigma_1 = [sigma_1 sigmahat1];    %append sigma into sigma_1 array
    mu_2 = [mu_2 muhat2];             %append mean into mu_2 array
    sigma_2 = [sigma_2 sigmahat2];    %append sigma into sigma_2 array
end

% plotting the graph to demonstrate the relationship
x = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999];
figure(1)
subplot(2,2,1)
plot(x, mu_1)      %plot no of samples vs mean in dataset1
xlabel('No of samples in 1st dataset')
ylabel('Mean')
title('No of samples vs Mean')
grid on

subplot(2,2,2)
plot(x, sigma_1)   %plot no of samples vs sigma in dataset1
xlabel('No of samples in 1st dataset')
ylabel('Covariance')
title('No of samples vs Covariance')
grid on

subplot(2,2,3)
plot(x, mu_2)      %plot no of samples vs mean in dataset2
xlabel('No of samples in 2nd dataset')
ylabel('Mean')
title('No of samples vs Mean')
grid on

subplot(2,2,4)
plot(x, sigma_2)   %plot no of samples vs sigma in dataset2
grid on
xlabel('No of samples in 2nd dataset')
ylabel('Covariance')
title('No of samples vs Covariance')

% Second part: Plot no of samples vs classification accuracy
X_data = [data1; data2];        %initialize input data by merging 2 classes of data
Y_data = csvread('target.csv', 1, 0);   %the class data for both dataset

prior = [0.9 0.1];         %specifying the priors
classNames = {'1','2'};     %defining the class names

% initialise array for classification accuracy for  data
loss_array=[];

for index = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 998]
    X = [X_data(1:index); X_data(1001:(1000+index))];
    Y = [Y_data(1:index); Y_data(1001:(1000+index))];
    Mdl = fitcnb(X,Y,'ClassNames',classNames,'Prior',prior);
    defaultPriorMdl = Mdl;
    defaultCVMdl = crossval(defaultPriorMdl);
    % Estimate the cross-validation error using 10-fold cross-validation.
    defaultLoss = kfoldLoss(defaultCVMdl); 
    loss_array = [loss_array defaultLoss];
end

figure(2)
x_num = [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 1998];
plot(x_num, loss_array)   %plot no of samples vs sigma in dataset2
grid on
xlabel('No of samples in the dataset')
ylabel('Loss')
title('No of samples vs Loss')

