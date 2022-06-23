import logging
from sacred import Experiment
import seml
import socket
import time
import sys
sys.path.append(".")
from utils.model_utils import *
import numpy as np
from verify_utils.gp_verify_utils import verify_single_image_gp

ex = Experiment()
seml.setup_logger(ex)


class MLP_50(nn.Module):
    def __init__(self):
        super(MLP_50,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784,57),
            nn.ReLU(),
            nn.Linear(57,45),
            nn.ReLU(),
            nn.Linear(45,46),
            nn.ReLU(),
            nn.Linear(46,52),
            nn.ReLU(),
            nn.Linear(52,49),
            nn.ReLU(),
            nn.Linear(49, 10),
            )
        self.output_dim=10
    def forward(self,x:torch.Tensor):
        x = x.view(x.shape[0],-1)
        return self.layers.forward(x)

def compare(arguments):

    device = arguments['device']
    eps = arguments['eps']
    weight_std = arguments['weight_std']
    # load test data
    train_loader, test_loader = find_right_model(
        DATASETS, arguments['data_set'],
        arguments=arguments,
        mean=arguments['mean'],
        std=arguments['std']
    )
    # create model and init model with different std
    model = MLP_50().to(device)
    for layer in model.layers:
        if isinstance(layer,nn.Linear):
            torch.nn.init.normal_(layer.weight,mean=0,std=weight_std)
    # verify a batch data
    imgs,labels = next(iter(test_loader))
    times=[]
    i=0

    for img in imgs:
        print(f"##### Processing image: {i} #####")
        start = time.time()
        verify_single_image_gp(model,img,eps=eps)
        end =time.time()
        i+=1
        times.append(end-start)
        print(f"spent {end-start}s")
    time_per_img = np.mean(times)

    return {"time_per_img":time_per_img}


def setup_gurobi():
    print('HOSTNAME: ', socket.gethostname())
    print(os.popen('hostid').read().strip())
    os.environ['GRB_LICENSE_FILE'] = '/nfs/homedirs/wangxun/gurobi_lic/{}.lic'.format(
        socket.gethostname())

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
    logging.info(f"dataset:{arguments['data_set']},  eps:{arguments['eps']}")
    logging.info(f"weight_std:{arguments['weight_std']}")
    setup_gurobi()
    
    results = compare(arguments)
    
    logging.info(f"running time: {results['time_per_img']}s per img")
    
    return results