import numpy as np
import cvxpy as cp 
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def verify_single_image(model,image,eps,label=None):
    """
    verify single image
    params:
    model: model to be verified
    image: input image
    eps: input perturbation budget
    """
    # define indicator: 0 for verified, 1 for not verified
    indicator = 0
    num_classes = model.output_dim
    if label==None:
    # predict unperturbed class
        c = model(image.reshape(1,-1)).argmax().item()
    else:
        c=label
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
    
    # get unstable_nums
    unstable_nums=[]
    # don't use the first layer and final layer
    for l,u in zip(L[1:-1],U[1:-1]):
        unstable_nums.append(int(((l<0) & (u>0)).sum()))

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
            if unstable_nums[i]==0:
                A.append(None)
            else:
                A.append(cp.Variable(shape=unstable_nums[i],boolean=True))
     
    # define Constraints for Liner layer and ReLU
    constraints = []
    for i in range(num_layers-1):
        # Linear for layer j
        if i==0:
            constraints += [X_hat[i+1]==W[i]@X_hat[i]+b[i]]
        else:
            constraints += [X_hat[i+1]==W[i]@Y[i-1]+b[i]]
        # ReLu for layer j
        k=0
        for j in range(shapes[i+1]):
            if L[i+1][j]>=0:
                constraints+=[Y[i][j]==X_hat[i+1][j]]
            elif U[i+1][j]<=0:
                constraints+=[Y[i][j]==0]
            else:
                constraints += [Y[i][j] <= X_hat[i+1][j] - L[i+1][j] * (1 - A[i][k]) ]
                constraints += [Y[i][j] >= X_hat[i+1][j] ]
                constraints += [Y[i][j] <= U[i+1][j] * A[i][k] ]
                constraints += [Y[i][j] >= 0 ]  
                k+=1     
    # define the problem
    # for other in tqdm(other_classes): #uncomment to show the progress bar
    for other in other_classes:    
        print(f"verifying label:{other}")
        problem = cp.Problem(objective=cp.Minimize(Y[-1][c]-Y[-1][other]),
                             constraints = constraints+[cp.atoms.norm_inf(image-X_hat[0]) <= eps,
                                                        Y[-1]== W[-1]@Y[-2]+b[-1]]
                             )
        opt = problem.solve('MOSEK')
        print(f"optimal:{opt}")
        if opt<0:
            print('found solution')
            indicator=1
            break
    return X_hat[0],Y[-1],indicator

# function used for multi processing
def verify_batch_images(model,imgs,labels,eps,total_num,failed_num,lock):
    """
    model should be in cpu
    imgs should be in cpu
    labels 
    total_num is a multiprocessing Value object to count the total number of imgs
    failed_num is a multiprocessing Value objec to count the total number of not verified imgs
    lock is used to prevent race condition
    """
    with lock:
        total_num.value+=len(imgs)
    for img,label in zip(imgs,labels):
        _,_,indicator = verify_single_image(model=model,image=img,label=label,eps=eps)
        if indicator==1:
            with lock:
                failed_num.value+=1

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
    for i in range(len(W)):
        if i==0 or i==len(W) : # first layer and last layer doesn't have relu
            U.append(W_p[i]@U[i]-W_m[i]@L[i]+b[i])
            L.append(W_p[i]@L[i]-W_m[i]@U[i]+b[i])
        else: # other layer has relu, apply relu to lower and upper bounds of previous layer
            U_=(U[i]>0)*U[i]
            L_=(L[i]>0)*L[i]
            U.append(W_p[i]@U_-W_m[i]@L_+b[i])
            L.append(W_p[i]@L_-W_m[i]@U_+b[i])
    return L,U