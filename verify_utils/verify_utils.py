import numpy as np
import cvxpy as cp 
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def verify_single_image(model,image,eps):
    """
    verify single image
    params:
    model: model to be verified
    image: input image
    eps: input perturbation budget
    """
    # predict unperturbed class
    num_classes = model.output_dim
    c = model(image.reshape(1,-1)).argmax().item()
    other_classes = set(range(num_classes))
    other_classes.remove(c)
    
    image = image.cpu().numpy().reshape(-1)
    
    # get the parameters 
    W,b = get_params_list(model)
    
    # get the layer_num
    num_layers = len(W)
    
    # get lower and upper bounds 
    L,U = get_lower_and_upper_bounds(image,eps,W,b)
    
    # get the shapes for each layer output
    shapes=[]
    for i in range(num_layers):
        shapes.append(U[i].shape[0])
    # add output shape
    shapes.append(num_classes)
    
    # define variables (X_hat,Y,A)
    # X_hat[0] is the input
    # Y[-1] is the output
    # need to have #num_layers-1 integer variables A for ReLU
    X_hat = []
    Y = []
    A = []
    for i in range(num_layers):
        X_hat.append(cp.Variable(shape=shapes[i]))
        Y.append(cp.Variable(shape=shapes[i+1]))
        if i != (num_layers-1):    
            A.append(cp.Variable(shape=shapes[i+1],boolean=True))
     
    # define Constraints for Liner layer and ReLU
    constraints = []
    for j in range(num_layers-1):
        # Linear for layer j
        if j==0:
            constraints += [X_hat[j+1]==W[j]@X_hat[j]+b[j]]
        else:
            constraints += [X_hat[j+1]==W[j]@Y[j-1]+b[j]]
        # ReLu for layer j
        constraints += [Y[j][i] <= X_hat[j+1][i] - L[j+1][i] * (1 - A[j][i]) for i in range(Y[j].shape[0])]
        constraints += [Y[j][i] >= X_hat[j+1][i] for i in range(Y[j].shape[0])]
        constraints += [Y[j][i] <= U[j+1][i] * A[j][i] for i in range(Y[j].shape[0])]
        constraints += [Y[j][i] >= 0 for i in range(Y[j].shape[0])]       
    # define the problem
    for other in tqdm(other_classes):
        problem = cp.Problem(objective=cp.Minimize(Y[-1][c]-Y[-1][other]),
                             constraints = constraints+[cp.atoms.norm_inf(image-X_hat[0]) <= eps,
                                                        Y[-1]== W[-1]@Y[-2]+b[-1]]
                             )
        opt = problem.solve('MOSEK')
        if opt<0:
            print('found solution')
            break
    return X_hat[0],Y[-1]

def get_params_list(model):
    """
    return a dict containing the model params 
    """
    W,b=[],[]
    for name,param in model.named_parameters():
        if "weight" in name:
            W.append(param.detach().cpu().numpy())     
        if "bias" in name:
            b.append(param.detach().cpu().numpy())
    return W,b

def get_lower_and_upper_bounds(x,eps,W,b):
    # construct W_+ and W_- for each layer
    W_p = []
    W_m = []
    for weight in W:
        W_p.append(np.maximum(0,weight))
        W_m.append(np.maximum(-weight,0))
    
    L = []
    U = []
    # l0 and u0
    L.append(x-eps)
    U.append(x+eps)
    # use interval arithmetic to get bounds for each layer
    for i in range(len(W)-1):
        U.append(W_p[i]@U[i]-W_m[i]@L[i]+b[i])
        L.append(W_p[i]@L[i]-W_m[i]@U[i]+b[i])        
    return L,U
        