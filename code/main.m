%% Example on how to use the ODE/PDE Solver: Solving  f'(x) = pi cos(pi x)  

% Define input dataset
clc; clear;
size_of_elements = 10000;
x = linspace(0,1,size_of_elements); 

% NN Initialisation
nn = nn_model('ODE Fitting using NN');
layers_dim = [1,3,1];
nn.initialize_parameters_deep(layers_dim,10);
nn.BC_coeff = 1;
nn.activation ='sigmoid';
RHS = pi * cos(pi * x);

% Train Network
nn.learning_rate = 0.01;
nn.num_iterations = 5000;
nn.train(x,RHS,false);

% Plotting Results
[yhat] = nn.predict(x, false);
plot(x,sin(pi * x),'LineWidth',2)
hold on
plot(x,yhat ,'LineWidth',2)
title('Regression using NN for $f(x) = sin(\pi x)$',"Interpreter","latex")
legend('Analytical Solution','NN Regression')

