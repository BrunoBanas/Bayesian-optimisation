import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import torch.distributions as dist

class acquisition():

    def __init__(self):
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight = 0.2     
        self.device = 'cuda:0'
        self.dtype = torch.float

    
    def empty_model(self):

        model_bnn = nn.Sequential(
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=22, out_features=300),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=300, out_features=400),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=400, out_features=1),
        nn.Flatten(0,1)
        )
        return model_bnn.to(self.device)

    def mu_std(self, X, X_sample, Y_sample, iterations_predict):

        X = X.reshape(-1, 22)
        
        #first step is to find the mean and std of the predictions - imitation of GP
        
        # Initialize a list to store the predictions for each iteration
        y_grid = []
        y_grid_sample = []
        
        for _ in range(iterations_predict): # retrain the model multiple times on the same dataset
            
            model_bnn_iter = self.empty_model()
            optimizer = torch.optim.RMSprop(model_bnn_iter.parameters(), lr=0.0001)

            #train the model on the few initial points 
            for step in range(300):
                    pre = model_bnn_iter(X_sample)
                    mse = self.mse_loss(pre, Y_sample)
                    kl = self.kl_loss(model_bnn_iter)
                    cost = mse + self.kl_weight*kl
                    
                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()
            #print('- MSE : %2.2f, KL : %2.2f' % (mse.item(), kl.item()))

            # Perform multiple iterations to obtain set of predictions
            for _ in range(15):
                
                y_predictions = model_bnn_iter(X) # predict the value at the suggested point X 
                y_sample_predictions = model_bnn_iter(X_sample)

                # Append the predictions to a list
                y_grid.append(y_predictions.tolist())    

                # Append the predictions to a list
                y_grid_sample.append(y_sample_predictions.tolist())
        
        # Convert the list of predictions to a PyTorch tensor
        sample_predictions_tensor = torch.tensor(y_grid_sample, dtype=self.dtype, device = self.device)

        # Calculate the mean and standard deviation along the iterations axis
        mu_sample = sample_predictions_tensor.mean(dim=0)

        # Convert the list of predictions to a PyTorch tensor
        predictions_tensor = torch.tensor(y_grid, dtype=self.dtype, device = self.device)

        # Calculate the mean and standard deviation along the iterations axis
        mu = predictions_tensor.mean(dim=0)
        sigma = predictions_tensor.std(dim=0)
        
        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        mu_sample_opt = mu_sample.max()   
        
        return mu, sigma, mu_sample_opt


    def expected_improvement_disc(self, X, X_sample, Y_sample, iterations_predict, xi):
        '''
        Computes the EI at points X based on existing samples X_sample
        and Y_sample using a surrogate model.
        
        Args:
            X: Points at which EI shall be computed (m x d).
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            xi: Exploitation-exploration trade-off parameter.
        
        Returns:
            Expected improvements at points X.
        '''
        
        mu, sigma, mu_sample_opt = self.mu_std(X, X_sample, Y_sample, iterations_predict)

        '''
        kappa = xi
        ucb = mu + kappa * sigma

        # Find the maximum value and its corresponding index along the dim=0 axis.
        max_ucb, max_idx = torch.max(ucb, dim=0)
        
        '''
        # Calculate the imp, mu, and sigma values for each point in the tensors.
        imp = mu - mu_sample_opt - xi

        # Use torch.where to handle the case when sigma is close to 0.
        # This ensures we don't run into division by zero issues.
        Z = torch.where(sigma < 1e-50, torch.tensor(0.0, device=self.device), imp / sigma) # sigma cutoff determines the amount of exploration

        # Use PyTorch functions for element-wise cdf and pdf calculations.
        ei = imp * dist.normal.Normal(0, 1).cdf(Z) + sigma * dist.normal.Normal(0, 1).log_prob(Z).exp()

        # Find the maximum value and its corresponding index along the dim=0 axis.
        max_ei, max_idx = torch.max(ei, dim=0)

        return max_idx