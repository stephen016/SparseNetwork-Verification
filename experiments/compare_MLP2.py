import logging
from sacred import Experiment
import numpy as np
import seml

import sys
sys.path.append(".")

from utils.model_utils import *
from utils.config_utils import *
from utils.system_utils import *

from verify_utils.verify_MLP2 import q_verify_single_image
from verify_utils.verify_utils import verify_single_image
import torch

import time
import timeit
import warnings


ex = Experiment()
seml.setup_logger(ex)

def compare(arguments):
    print("get device")
    device = arguments['device']
    # load model
    checkpoint_name = arguments['checkpoint_name']
    checkpoint_MLP2_model = arguments['checkpoint_model1']
    model_path = os.path.join(RESULTS_DIR,checkpoint_name,MODELS_DIR,checkpoint_MLP2_model)
    model_MLP=DATA_MANAGER.load_python_obj(model_path,device='cpu')
    
    checkpoint_qMLP2_model = arguments['checkpoint_model2']
    model_path = os.path.join(RESULTS_DIR,checkpoint_name,MODELS_DIR,checkpoint_qMLP2_model)
    model_qMLP=DATA_MANAGER.load_python_obj(model_path,device='cpu')
    print("load model finished")

    configure_seeds(arguments,device)

    # load dataset
    train_loader, test_loader = find_right_model(
        DATASETS, arguments['data_set'],
        arguments=arguments,
        mean=arguments['mean'],
        std=arguments['std']
    )
    print("get data")
    eps = arguments['eps']
    loop = arguments['loop']
    # choose only the first batch as compare data (128 images)
    # report accuracy and adversarial accuracy and running time
    print("get settings")
    print(f"total_batch_num= {len(test_loader)}")
    i=0
    
    x,y = next(iter(test_loader))
    
    print("y",y)
    #print(labels)
    total_num = arguments['batch_size']
    print("get batch_size")
    # get the results from original model
    # accuracy
    acc_MLP = get_accuracy(model=model_MLP,images=x,labels=y)
    print("get acc",acc_MLP)
    # adversarial accuracy
    failed_num_MLP=verify_MLP2(model=model_MLP,images=x,eps=eps)
    adv_acc_MLP = (total_num-failed_num_MLP)/total_num
    total_time_MLP = timeit.timeit(lambda: verify_MLP2(model_MLP,x,eps), number=loop)
    # time per image
    time_MLP = total_time_MLP/(loop*total_num)
    # get the resuts from quantized model
    acc_qMLP = get_accuracy(model=model_qMLP,images=x,labels=y)
    failed_num_qMLP=verify_MLP2(model=model_qMLP,images=x,eps=eps)
    adv_acc_qMLP = (total_num-failed_num_qMLP)/total_num
    total_time_qMLP = timeit.timeit(lambda: verify_MLP2(model_qMLP,x,eps), number=loop)
    # time per image
    time_qMLP = total_time_qMLP/(loop*total_num)

    return acc_MLP,adv_acc_MLP,time_MLP,acc_qMLP,adv_acc_qMLP,time_qMLP

def get_accuracy(model,images,labels):
    print("start running")
    pred=model(images)
    pred=torch.argmax(pred,axis=1)
    return torch.sum(labels==pred).item()/len(labels)
def verify_MLP2(model,images,eps):
    failed_num = 0
    for img in images:
        if hasattr(model,"quant"): # quantized model
            _,_,indicator = q_verify_single_image(model=model,image=img,eps=eps)
        else:
            _,_,indicator = verify_single_image(model=model,image=img,eps=eps)
        failed_num+=indicator
    return failed_num
        


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(arguments):
    warnings.filterwarnings("ignore")
    logging.info(f"using checkpoint f{arguments['checkpoint_name']}")
    logging.info('Received the following configuration:')
    logging.info(f"dataset:{arguments['data_set']}, eps:{arguments['eps']}")
    
    acc_MLP,adv_acc_MLP,time_MLP,acc_qMLP,adv_acc_qMLP,time_qMLP = compare(arguments)

    logging.info('MLP model')
    logging.info(f"accuracy: {acc_MLP}, adversarial accuracy: {adv_acc_MLP}, running time:{time_MLP}s per image")
    
    logging.info('quantized MLP model')
    logging.info(f"accuracy: {acc_qMLP}, adversarial accuracy: {adv_acc_qMLP}, running time:{time_qMLP}s per image")
    