import logging
from sacred import Experiment
import numpy as np
import seml

import sys
sys.path.append(".")

from utils.model_utils import *
from utils.config_utils import *
from utils.system_utils import *

from verify_utils.verify_utils import verify_batch_images
from multiprocessing import Process, Value, Lock
import time
import timeit

ex = Experiment()
seml.setup_logger(ex)

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
    num_process = arguments['num_process']
    eps = arguments['eps']
    lock=Lock()
    total_num = Value('i',0)
    failed_num = Value('i',0)
    total_batch_num=len(test_loader)
    batch_num=1
    for imgs,labels in test_loader: # iterate over all test data
        print(f"verifying batch {batch_num}/{total_batch_num}")
        # using only one batch
        print("only verify 1st batch")
        if batch_num >1:
            break
        batch_num+=1
        labels=model(imgs).argmax(axis=1).numpy()
        processes=[]
        num_per_process=int(len(imgs)/num_process)
        for i in range(num_process):
            if i!=num_process-1:
                process_imgs=imgs[num_per_process*i:num_per_process*(i+1)].squeeze()
                process_labels = labels[num_per_process*i:num_per_process*(i+1)]
            else:
                process_imgs=imgs[num_per_process*i:].squeeze()
                process_labels = labels[num_per_process*i:]
            p=Process(target=verify_batch_images,args=(model,process_imgs,process_labels,eps,total_num,failed_num,lock))
            processes.append(p)
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        
    return {"total_num":total_num.value, "failed_num":failed_num.value}

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
    logging.info(f"dataset:{arguments['data_set']}, model:{arguments['checkpoint_model']}, eps:{arguments['eps']},num_process:{arguments['num_process']}")
    
    loop = 1
    total_time = timeit.timeit(lambda: verify(arguments), number=loop)
    results = verify(arguments)
    
    logging.info(f"total_num:{results['total_num']}, failed_num:{results['failed_num']}")
    logging.info(f"adversarial accuracy: {(results['total_num']-results['failed_num'])/results['total_num']}")
    print(f"num of process: {arguments['num_process']}")
    logging.info(f"average running time: {total_time/loop}s")
    
    return results