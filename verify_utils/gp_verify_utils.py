import gurobipy as gp
from gurobipy import GRB
from verify_utils.verify_utils import get_params_list,get_lower_and_upper_bounds

def terminate_early(model, where):
    if where == GRB.Callback.MIP:
        #runtime = model.cbGet(GRB.Callback.RUNTIME)
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        if objbnd>0 and objbst>0:
            model.terminate()


def verify_single_image_gp(model,image,eps):
    num_classes = model.output_dim
    
    c = model(image.reshape(1,-1)).argmax().item()

    other_classes = set(range(num_classes))
    other_classes.remove(c)

    img = image.cpu().numpy().reshape(-1)
    
    # get the parameters 
    W,b = get_params_list(model)
    # get the lower and upper bounds
    L,U = get_lower_and_upper_bounds(img,eps,W,b)

    # get the layer_num
    num_layers = len(W)
    # get shapes of every layer 
    shapes=[]
    for i in range(num_layers):
        shapes.append(U[i].shape[0])
    shapes.append(num_classes)


    unstable_nums=[]
    # don't use the first layer and final layer
    for l,u in zip(L[1:-1],U[1:-1]):
        unstable_nums.append(int(((l<0) & (u>0)).sum()))
    

    # create model 
    m = gp.Model("verify-n-layer")
    # create variables
    X=[]
    Y=[]
    A=[]
    # X[0] is input X[-1] is output
    # Y is output after relu
    # A is binary variable for relu constraints
    for i in range(num_layers+1):
        X.append(m.addMVar((shapes[i]),lb=L[i],ub=U[i],name=f"x_{i}"))
    for i in range(num_layers-1):
        Y.append(m.addMVar(shapes[i+1],lb=0,name=f"y_{i}"))
        
    
    for i,num in enumerate(unstable_nums):
        A.append(m.addVars(num, vtype=GRB.BINARY,name=f"a_{i}"))
   
    # define constraints
    # input perturbation already defined as lower and upper bounds on X_0
    #m.addConstrs((X[0][i]>=img[i]-eps for i in range(shapes[0])));
    #m.addConstrs((X[0][i]<=img[i]+eps for i in range(shapes[0])));
    # Linear 
    # X1 = W0 @ X0 + b0
    # X2 = W1 @ Y0 + b1
    # X3 = W2 @ Y1 + b2
    # ...
    for i in range(num_layers):
        if i==0:
            m.addConstr(W[0]@X[0]+b[0]==X[1])
        else:
            m.addConstr(W[i]@Y[i-1]+b[i]==X[i+1])
    """
    relu 
        if stable(l>0 y0=x1, u<0, y0=1)
        if unstable (y0=relu(x1))
        
    # Y0 = relu(X1)
    # Y1 = relu(X2)
    """
    for i in range(num_layers-1):
        k=0
        for j in range(shapes[i+1]):
            if L[i+1][j]>=0:
                m.addConstr(Y[i][j]==X[i+1][j]);
            elif U[i+1][j]<=0:
                m.addConstr(Y[i][j]==0);
            else:
                m.addConstr(Y[i][j]<=X[i+1][j]-L[i+1][j]*(1-A[i][k])) ;
                m.addConstr(Y[i][j]<=U[i+1][j]*A[i][k]);
                m.addConstr(Y[i][j]>=X[i+1][j]);
                m.addConstr(Y[i][j]>=0);
                k+=1
    m.Params.LogToConsole = 0

    infeasible=0
    verified = 1
    for other in other_classes:
        print(f"verifying label: {other}")
        m.setObjective(X[-1][c]-X[-1][other], GRB.MINIMIZE)
        #m.optimize()
        # terminate early
        m.optimize(terminate_early)
        try:
            if m.ObjVal<0:
                verified=0
                infeasible=0
                print("found solution")
                break
        except AttributeError:   
            infeasible=1
            continue
    if infeasible==1:
        verified=0
    
    return verified,infeasible