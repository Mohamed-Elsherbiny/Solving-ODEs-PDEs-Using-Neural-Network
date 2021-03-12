    %
%============================================================================
 % Name : DiffEq
 % Author : Mohamed Elsherbiny
 % Version : ver 1.0
 % Copyright : For free
 % Description : Function to define the ODEs/PDEs for nn_model class
 % Note : You have to write your diffrential equation here 
 %
%===========================================================================
function [BC,DiffTerm] = DiffEq(x,fx,dfxdx,dfx2d2x)
%       Arguments:
%       x       -- input data for training. size (n,m), n is the number of variables/features, m is the nuber of training data 
%       fx      -- output of the neural network f(x). size(1,m)
%       dfxdx   -- first derivative of f(x). size(n,m)
%       dfx2d2x -- second derivative of f(x). size(n,m)

%       Returns:
%       BC       -- Boundary Conditions
%       DiffTerm -- Diffrential equation part

%--------------------------------------------------------------------
%-----------------FIRST-ORDER-ODE: df/dx = pi * cos(pi * x)----------
%--------------------------------------------------------------------
% m = length(fx);
% DiffTerm = dfxdx;
% 
% zero = zeros(1,m-1);
% BC = [(fx(1) - 0) zero];

%--------------------------------------------------------------------
%-----------------FIRST-ORDER-ODE: df/dx + f(x) = exp(-x).*cos(x)----
%--------------------------------------------------------------------
m = length(fx);
DiffTerm = dfxdx + fx;

zero = zeros(1,m-1);
BC = [(fx(1) - 0) zero];


end

