import logging
from sacred import Experiment
import numpy as np
import seml
import socket

import sys
sys.path.append(".")

from utils.model_utils import *
from utils.config_utils import *
from utils.system_utils import *

from verify_utils.gp_verify_utils import verify_single_image_gp

import time
import timeit


ex = Experiment()
seml.setup_logger(ex)

def setup_gurobi():
    print('HOSTNAME: ', socket.gethostname())
    print(os.popen('hostid').read().strip())
    os.environ['GRB_LICENSE_FILE'] = '/nfs/homedirs/wangxun/gurobi_lic/{}.lic'.format(
        socket.gethostname())


def verify(arguments):
    device = arguments['device']
    # load model
    checkpoint_name = arguments['checkpoint_name']
    checkpoint_model = arguments['checkpoint_model']
    model_path = os.path.join(RESULTS_DIR,checkpoint_name,MODELS_DIR,checkpoint_model)
    model=DATA_MANAGER.load_python_obj(model_path,device='cpu').to(device)

    configure_seeds(arguments,device)
    # load dataset
    train_loader, test_loader = find_right_model(
        DATASETS, arguments['data_set'],
        arguments=arguments,
        mean=arguments['mean'],
        std=arguments['std']
    )
    eps = arguments['eps']
    # get the model accuracy
    correct=0
    total=0
    for imgs,labels in test_loader:
        pred=torch.argmax(model(imgs),axis=1)
        correct+=torch.sum(pred==labels).item()
        total+=len(labels)
    acc = correct/total
    print(f"model accuracy: {acc}")

    # only verify one batch data
    print("verify first batch data")
    imgs,labels = next(iter(test_loader))
    verified_num = 0
    infeasible_num = 0
    correct_num=0
    data_num=len(labels)
    
    i=0
    incorrect=[]
    unverified=[]
    times=[]
    for (img,label) in zip(imgs,labels):
        print(f"##### Processing image: {i} #####")
        # only verify correctly predicted imgs
        pred=torch.argmax(model(img),axis=1)
        if pred == label:
            correct_num+=1
            print("correct classified, process verify")
            # only count time for processing verification
            start=time.time()
            verified,infeasible = verify_single_image_gp(model,img,eps)
            end = time.time()
            times.append(end-start)
            verified_num+=verified
            infeasible_num+=infeasible
            if not verified:
                unverified.append(i)
        else:
            print("incorrect classified, skip verify")
            incorrect.append(i)

        i+=1
    time_per_img = np.mean(times)
    
    return {"accuracy":acc,"data_num":data_num,"correct_num":correct_num,"verified_num":verified_num, "infeasible_num":infeasible_num,
            "time_per_img":time_per_img}

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
    logging.info('Received the following configuration:')
    logging.info(f"dataset:{arguments['data_set']}, model:{arguments['checkpoint_model']}, eps:{arguments['eps']}")
    setup_gurobi()
    
    results = verify(arguments)
    logging.info(f"model accuracy:{results['accuracy']}")
    logging.info(f"test_verify_num:{results['data_num']}, correct_num:{results['correct_num']},verified_num:{results['verified_num']}")
    logging.info(f"infeasible_num: {results['infeasible_num']}")
    logging.info(f"adversarial accuracy: {results['verified_num']/(results['correct_num']-results['infeasible_num'])}")
    logging.info(f"running time: {results['time_per_img']}s per img")
    
    return results