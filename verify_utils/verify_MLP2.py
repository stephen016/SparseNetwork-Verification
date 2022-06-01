import numpy as np
import cvxpy as cp 


def q_verify_single_image(model,image,eps):
    """
    verify single image
    params:
    model: model to be verified now only support MLP2
    image: input image
    eps: input perturbation budget
    """
    indicator = 0
    # get num_classes and model output
    num_classes=model.output_dim
    #c = torch.argmax(model(image)).item()
    c = model(image.reshape(1,-1)).argmax().item()
    other_classes = set(range(num_classes))
    other_classes.remove(c)
    
    # get the fixed parameters of the model
    
    W_int,W_scales,B,out_scales,out_zps,in_scale,in_zp = get_parameter_list(model)
    
    # define parameters
    W1=W_int[0]
    b1=B[0]
    W2=W_int[1]
    b2=B[1]
    # change the shape of image
    img=image.numpy().reshape(-1)
    
    # calculate lower and upper bound
    # lower and upper bounds before first quantization 
    u0=img+eps
    l0=img-eps
    # get lower and upper bounds after scaling
    in_scale_p = np.maximum(0,1/in_scale)
    in_scale_m = np.maximum(-1/in_scale,0)

    u0_1 = in_scale_p*u0-in_scale_m*l0+in_zp
    l0_1 = in_scale_p*l0-in_scale_m*u0+in_zp

    # get lower and upper bounds after round
    _u0_int=np.round(u0_1)
    _l0_int=np.round(l0_1)

    # get lower and upper bounds after clamp
    u0_int=np.clip(_u0_int,a_min=0,a_max=None)
    l0_int=np.clip(_l0_int,a_min=0,a_max=None)

    # get lower and upper bounds after first layer calculation
    W1_p = np.maximum(0,W1)
    W1_m = np.maximum(-W1,0)

    u1 = ((W1_p@u0_int - W1_m@l0_int)-W1@in_zp.repeat(u0_int.shape[0]))*W_scales[0]*in_scale/out_scales[0] + b1/out_scales[0]+out_zps[0]
    l1 = ((W1_p@l0_int - W1_m@u0_int)-W1@in_zp.repeat(u0_int.shape[0]))*W_scales[0]*in_scale/out_scales[0] + b1/out_scales[0]+out_zps[0]


    # get lower and upper bounds after round
    _u1_int=np.round(u1)
    _l1_int=np.round(l1)

    # get lower and upper bounds after clamp
    u1_int=np.clip(_u1_int,a_min=0,a_max=None)
    l1_int=np.clip(_l1_int,a_min=0,a_max=None)

    # get lower and upper bounds after relu  do I need this ?

    # get lower and upper bounds after second layer calculation
    # the output scale and zero point of first layer are the input scale and zero point of second layer
    W2_p = np.maximum(0,W2)
    W2_m = np.maximum(-W2,0)

    u2 = ((W2_p@u1_int - W2_m@l1_int)-W2@np.array(out_zps[0]).repeat(u1_int.shape[0]))*W_scales[1]*out_scales[0]/out_scales[1] + b2/out_scales[1]+out_zps[1]
    l2 = ((W2_p@l1_int - W2_m@u1_int)-W2@np.array(out_zps[0]).repeat(u1_int.shape[0]))*W_scales[1]*out_scales[0]/out_scales[1] + b2/out_scales[1]+out_zps[1]

    # get lower and upper bounds after round
    _u2_int=np.round(u1)
    _l2_int=np.round(l1)
    
    #define variables
    # step 1 quantize x
    # _x = x/in_scale + in_zp
    # _x_int = round(_x)
    # x_int = clamp(_x_int,min=0)
    x = cp.Variable(shape=(img.shape))
    _x = cp.Variable(shape=(img.shape))
    _x_int = cp.Variable(shape=(img.shape),integer=True)
    x_int = cp.Variable(shape=(img.shape),integer=True)
    # boolean variable for clamp constraint
    a0 = cp.Variable(shape=(img.shape), boolean=True)


    # output of first layer
    # y1 = W1@(x_int-x_zp)*W_scales[0]*in_scale/out_scales[0]+b1/out_scales[0]+out_zps[0]
    # _y1_int = round(y1)
    # y1_int = clamp(_y1_int,min=0)
    y1 = cp.Variable(shape=(W1.shape[0]))
    _y1_int = cp.Variable(shape=(W1.shape[0]),integer=True)
    y1_int = cp.Variable(shape=(W1.shape[0]),integer=True)
    # boolean variable for clamp constraints
    a1_1 = cp.Variable(shape=(W1.shape[0]), boolean=True)

    # output of a ReLU function
    y1_out = cp.Variable(shape=(W1.shape[0]),integer=True)
    #  boolean variable for relu constraints
    a1_2 = cp.Variable(shape=(W1.shape[0]), boolean=True)

    # output of second layer
    # y2 = W2@(y1_out-out_zps[0])*W_scales[1]*out_scales[0]/out_scales[1]+b2/out_scales[1]+out_zps[1]
    # _y2_int = round(y2)
    # y2_int = clamp(_y2_int,min=0)
    y2 = cp.Variable(shape=(W2.shape[0]))
    _y2_int = cp.Variable(shape=(W2.shape[0]),integer=True)
    y2_int = cp.Variable(shape=(W2.shape[0]),integer=True)
    #  boolean variable for clamp constraints
    a2 = cp.Variable(shape=(W2.shape[0]), boolean=True) 
    
    
    ## define constraints
    # define constraints
    constraints=[]
    constraints += [_x==x/in_scale+in_zp]
    # round operation
    constraints += [cp.atoms.norm_inf(_x-_x_int) <= 0.5]
    # clamp operation one similar to relu 
    # constrains of clip (x>0)
    constraints += [x_int[i] <= _x_int[i] - _l0_int[i] * (1 - a0[i]) for i in range(x_int.shape[0])]
    constraints += [x_int[i] >= _x_int[i] for i in range(x_int.shape[0])]
    constraints += [x_int[i] <= _u0_int[i]*a0[i] for i in range(x_int.shape[0])]
    constraints += [x_int[i] >= 0 for i in range(x_int.shape[0])]

    # constraints of first layer 
    constraints += [y1 == W1@(x_int-in_zp)*W_scales[0]*in_scale/out_scales[0]+b1/out_scales[0]+out_zps[0]]

    # constrains of round operation
    constraints += [cp.atoms.norm_inf(y1-_y1_int) <= 0.5]

    # constrains of clip
    constraints += [y1_int[i] <= _y1_int[i] - _l1_int[i] * (1 - a1_1[i]) for i in range(y1_int.shape[0])]
    constraints += [y1_int[i] >= _y1_int[i] for i in range(y1_int.shape[0])]
    constraints += [y1_int[i] <= _u1_int[i]*a1_1[i] for i in range(y1_int.shape[0])]
    constraints += [y1_int[i] >= 0 for i in range(y1_int.shape[0])]
    
    # constraints for relu of first layer (zero_point = out_zps[0])
    constraints += [y1_out[i] <= y1_int[i] - (l1_int[i]-out_zps[0]) * (1 - a1_2[i]) for i in range(y1_int.shape[0])]
    constraints += [y1_out[i] >= y1_int[i] for i in range(y1_int.shape[0])]
    constraints += [y1_out[i] <= out_zps[0]+(u1_int[i]-out_zps[0])*a1_2[i] for i in range(y1_int.shape[0])]
    constraints += [y1_out[i] >= out_zps[0] for i in range(y1_int.shape[0])]

    # constraints of second layer
    constraints += [y2 == W2@(y1_out-out_zps[0])*W_scales[1]*out_scales[0]/out_scales[1]+b2/out_scales[1]+out_zps[1]]

    # constraints of round operation
    constraints += [cp.atoms.norm_inf(y2-_y2_int) <= 0.5]

    # constraints of clip
    constraints += [y2_int[i] <= _y2_int[i] - _l2_int[i] * (1 - a2[i]) for i in range(y2_int.shape[0])]
    constraints += [y2_int[i] >= _y2_int[i] for i in range(y2_int.shape[0])]
    constraints += [y2_int[i] <= _u2_int[i]*a2[i] for i in range(y2_int.shape[0])]
    constraints += [y2_int[i] >= 0 for i in range(y2_int.shape[0])]

    # constrainst of input pertubation
    constraints += [cp.atoms.norm_inf(img-x) <= eps]
    
    for other in other_classes:
        problem=cp.Problem(objective=cp.Minimize(y2_int[c]-y2_int[other]),constraints=constraints)
        opt = problem.solve('MOSEK')
        if opt < 0:
            print('found solution')
            indicator=1
            break
    return x,y2_int,indicator
def get_parameter_list(model):
    W_scales=[]
    W_int=[]
    B=[]
    out_scales=[]
    out_zps=[]
    in_scale = model.quant.scale.numpy()
    in_zp = model.quant.zero_point.numpy()

    for layer in model.layers:
        if hasattr(layer,"scale"):
            w,b = layer._weight_bias()
            W_scales.append(w.q_per_channel_scales().numpy())
            W_int.append(w.int_repr().numpy())
            B.append(b.detach().numpy().reshape(-1))
            out_scales.append(layer.scale)
            out_zps.append(layer.zero_point)
    return W_int,W_scales,B,out_scales,out_zps,in_scale,in_zp

