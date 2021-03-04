classdef nn_model <handle
    %
%============================================================================
 % Name : nn_model
 % Author : Mohamed Elsherbiny
 % Version : ver 1.0
 % Copyright : For free
 % Description : Neural Netwrok Class for solving ODEs/PDEs
 % Note : -
 %
%===========================================================================
    
    properties
        name
        layer_dims
        cost
        W
        B
        num_iterations
        learning_rate
        itr
        d_dx
        caches
    end
    
    methods
        %% Constructor
        function obj = nn_model(val)
            if nargin == 1
                obj.name = val;
            end
        end
        %% Initialization
        function [W, b] = initialize_parameters_deep(obj, layer_dims)
    
%         Arguments:
%         layer_dims -- array containing the dimensions of each layer in our network
% 
%         Returns:
%         parameters -- containing your parameters "W1", "b1", ..., "WL", "bL":
%                         Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
%                         bl -- bias vector of shape (layer_dims[l], 1)

            L = length(layer_dims);            % number of layers in the network
            obj.layer_dims = L;
            for L = 2:L
                W{L-1} = rand(layer_dims(L),layer_dims(L-1))*8 ;
                b{L-1} = ones(layer_dims(L),1)*0.01;
            end
            obj.W = W;
            obj.B = b;
        end
        %% Linear Forward
        function [Z, cache] = linear_forward(obj, A, W, b)
%     
%       Implement the linear part of a layer's forward propagation.
% 
%       Arguments:
%       A -- activations from previous layer (or input data): (size of previous layer, number of examples)
%       W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
%       b -- bias vector, numpy array of shape (size of the current layer, 1)
% 
%       Returns:
%       Z -- the input of the activation function, also called pre-activation parameter 
%       cache -- "A", "W" and "b" ; stored for computing the backward pass efficiently
        Z = W*A + b;
        cache = {A, W, b};
        end
        %% Linear-Activation Forward
        function [A, cache] = linear_activation_forward(obj, A_prev, W, b, activation)
%       Implement the forward propagation for the LINEAR->ACTIVATION layer

%       Arguments:
%       A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
%       W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
%       b -- bias vector, numpy array of shape (size of the current layer, 1)
%       activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
   
%       Returns:
%       A -- the output of the activation function, also called the post-activation value 
%       cache -- containing "linear_cache" and "activation_cache" stored for computing the backward pass efficiently    
            if activation == "sigmoid"
                [Z, linear_cache] = obj.linear_forward(A_prev, W, b);
                A = obj.sigmoid(Z);
                cache = [linear_cache, Z];
            end

        end
        %% L-Layer Model
        function [AL, caches] = L_model_forward(obj, X, W,b)
%       Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
%       Arguments:
%       X -- data, numpy array of shape (input size, number of examples)
%       parameters -- output of initialize_parameters_deep()
    
%       Returns:
%       AL -- last post-activation value
%       caches -- list of caches containing:
%       every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
            A = X;
            L = obj.layer_dims;                 % number of layers in the neural network
            caches = {};
            % Implement [LINEAR -> SIGMOID]*(L-1). Add "cache" to the "caches" list.
            for l = 1:L-2
                A_prev = A ;
                [A, linear_activation_cache] = obj.linear_activation_forward(A_prev,W{l},b{l},'sigmoid');
                caches = [caches linear_activation_cache];
            end
            
            % Implement LINEAR @ output layer
            [AL, linear_cache] = obj.linear_forward(A, W{L-1}, b{L-1});
            caches = {caches linear_cache};
            obj.caches = caches;
        end


        %% Cost Function
        function cost = loss(obj,AL,Y)

%       Arguments:
%       AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
%       Y -- true "label" vector , shape (1, number of examples)

%       Returns:
%       cost -- MSE
            m = length(Y);
            cost = 1/m * sum((Y - AL).^2);
            % dA = -2/m * cost  
        end
        %% implement sigmoid_backward
        function [dZ] = sigmoid_backward(obj,dA, activation_cache)
            %       Implements the backward propagation for SIGMOID unit.
            
            %       Arguments:
            %       dA -- post-activation gradient for current layer l 
            %       activation_cache -- (A_prev, W, b) coming from the forward propagation in the current layer
            
            %       Returns:
           %       dZ -- Gradient of the cost with respect to z (current layer l), same shape as z
                    
            Z = activation_cache;
            dZ = dA .* obj.sigmoid_diff(Z);
        end
            


        %% implement linear_backward
        function [dA_prev, dW, db] = linear_backward(obj,dZ, cache)
%       Implement the linear portion of backward propagation for a single layer (layer l)

%       Arguments:
%       dZ -- Gradient of the cost with respect to the linear output (of current layer l)
%       cache -- (A_prev, W, b) coming from the forward propagation in the current layer

%       Returns:
%       dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
%       dW -- Gradient of the cost with respect to W (current layer l), same shape as W
%       db -- Gradient of the cost with respect to b (current layer l), same shape as b
            [A_prev, W, b] = cache{1:3};
            m = length(A_prev(1,:));

            dW =  dZ *  A_prev';
            db = sum(dZ,2) ;
            dA_prev = W' * dZ;
            %dA_prev = 0;
        end


        %% Implement the backpropagation for the LINEAR->ACTIVATION layer.
        function [dA_prev, dW, db] = linear_activation_backward(obj,dA, cache, activation)
%           Arguments:
%       dA -- post-activation gradient for current layer l 
%       cache -- (linear_cache, activation_cache) we store for computing backward propagation efficiently
%       activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
%       Returns:
%       dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
%       dW -- Gradient of the cost with respect to W (current layer l), same shape as W
%       db -- Gradient of the cost with respect to b (current layer l), same shape as b     
            linear_cache = {cache{1:3}};
            activation_cache = cache{4};
            if activation == 'sigmoid'
                dZ = obj.sigmoid_backward(dA, activation_cache);
                [dA_prev, dW, db] = obj.linear_backward(dZ, linear_cache);
            end


        end

        %% L_model_backward
        function [grad] = L_model_backward(obj,AL, Y, caches)
%       Arguments:
%       AL -- probability vector, output of the forward propagation (L_model_forward()) 
%       Y -- true "label" vector 
%       caches -- list of caches containing:
%       every cache of linear_activation_forward() with "sigmoid" (it's caches[l], for l in range(L-1) i.e l = 0...L-1)
%       the cache of linear_activation_forward() with "Linear" (it's caches[L])

%       Returns:
%       grads -- dA, dW and db
%           
            m = length(Y);
            cost = obj.loss(AL,Y);

            dAL = -2/m * (Y-AL);
            L = obj.layer_dims;
            grad = {};
            % Lth layer
            current_cache = caches{end} ;
            linear_cache = {current_cache{1:3}};
            [dA_prev_temp, dW_temp, db_temp] = obj.linear_backward(dAL, linear_cache) ;
            %grad = {grad, {dA_prev_temp, dW_temp, db_temp}};
            grad = {grad, {dW_temp, db_temp}}; 
            grad = grad(~cellfun('isempty',grad));
            grad{L-1} = grad{1};
            
            % L-1 Layers
            for i = L-2:-1:1
                current_cache = {caches{1}{4*i-3:4*i}} ;
                [dA_prev_temp, dW_temp, db_temp] = obj.linear_activation_backward(dA_prev_temp, current_cache, 'sigmoid') ;
                %grad = {{dA_prev_temp, dW_temp, db_temp}, grad};
                grad{i} = {dW_temp, db_temp}; 
            end

        end

        %% Update Parameters
        function [W, B] = update_parameters(obj, parameters, grads,learning_rate)
%       Update parameters using gradient descent
%       Arguments:
%       parameters --  containing your parameters: List {W;B}
%       grads -- containing your gradients, output of L_model_backward
%       
%       Returns:
%       parameters -- containing your updated parameters 
            L = obj.layer_dims;                 % number of layers in the neural network
            W = parameters{1};
            B = parameters{2};
            
            
            for l = 1:L-1
                W{1,l} = W{1,l} - learning_rate .* reshape(grads{1,l}{1},size(W{1,l}));
                B{1,l} = B{1,l} - learning_rate .* grads{1,l}{2}; 
            end
            
        end
        
        %% train
        function [W, B] = train(obj, input_data, output,print_cost)
            %       train parameters using gradient descent
            %       Arguments:
            %       input_data --  containing your input data set
            %       output -- containing your desired output
            %       
            %       Returns:
            %       parameters -- containing your updated parameters 

            if nargin<3
                print_cost = false;
            end
            w = obj.W;
            b = obj.B;
            itr = obj.num_iterations;
            parameters = {w;b};
            learning_rate = obj.learning_rate;
            


            % training loop
            for i = 1: itr
                % 1- forward propagation
                [Al, caches] = obj.L_model_forward(input_data, w, b);
                % 2- Compute Cost
                cost = obj.loss(output,Al);
                % 3- Backward propagation
                grads = obj.L_model_backward(Al, output, caches);
                % 4- Update Parameters
                [w, b] = obj.update_parameters(parameters, grads, learning_rate);
                parameters = {w;b};
                if print_cost
                    fprintf('cost = %f \n',cost)
                end
                if cost < 0.001
                    obj.itr = itr;
                    break;
                end
                obj.itr = itr;
            end
            obj.W = w;
            obj.B = b;
            obj.cost = cost;
        end

        function gradient_X(obj,AL)
%       Arguments:
%       AL -- probability vector, output of the forward propagation (L_model_forward()) 
%       caches -- list of caches containing:
%       every cache of linear_activation_forward() with "sigmoid" (it's caches[l], for l in range(L-1) i.e l = 0...L-1)
%       the cache of linear_activation_forward() with "Linear" (it's caches[L])

%       Returns:
%       void
%           
            m = length(AL);
            dAL = ones(1,m);
            L = obj.layer_dims;
            caches = obj.caches;
            % Lth layer
            current_cache = caches{end} ;
            linear_cache = {current_cache{1:3}};
            [dA_prev_temp, dW_temp, db_temp] = obj.linear_backward(dAL, linear_cache) ;
            obj.d_dx{L-1} = {dA_prev_temp};

            % L-1 Layers
            for i = L-2:-1:1
                current_cache = {caches{1}{4*i-3:4*i}} ;
                [dA_prev_temp, dW_temp, db_temp] = obj.linear_activation_backward(dA_prev_temp, current_cache, 'sigmoid') ;
                obj.d_dx{i} = {dA_prev_temp};
            end

        end

        %%
        function z = sigmoid(obj, x)
        z = 1 ./ (1 + exp(-x));
        end
        
        %% implement linear_backward
        function [dg] = sigmoid_diff(obj, Z)
            
            dg = obj.sigmoid(Z) .* (1 - obj.sigmoid(Z));
            
            
        end
    end
end

