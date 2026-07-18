from utils import tab_printer
from trainer import Trainer
from param_parser import parameter_parser
import random
import numpy as np
import os
import torch

def seed_everything(TORCH_SEED):
	random.seed(TORCH_SEED)
	os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
	np.random.seed(TORCH_SEED)
	torch.manual_seed(TORCH_SEED)
	torch.cuda.manual_seed_all(TORCH_SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def main():
    
    seed_everything(0)
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a PartialDiffGED model.
    """
    args = parameter_parser()
    if int(os.environ.get("RANK", "0")) == 0:
        tab_printer(args)

    trainer = Trainer(args)

    if args.model_epoch_start > 0:
        trainer.load(args.model_epoch_start)

    if args.model_train == 1:
        for epoch in range(args.model_epoch_start, args.model_epoch_end):
            trainer.cur_epoch = epoch
            trainer.fit()
            completed_epochs = epoch + 1
            if args.save_every_epochs > 0 and completed_epochs % args.save_every_epochs == 0:
                trainer.save(completed_epochs)
        if args.model_epoch_end == 0 or args.save_every_epochs <= 0 or args.model_epoch_end % args.save_every_epochs != 0:
            trainer.save(args.model_epoch_end)

        #trainer.score('test',test_k=100,top_k_approach='parallel')
        
    else:
        if args.experiment == 'test':
            trainer.score(testing_graph_set=args.testset,test_k=args.test_k,top_k_approach=args.topk_approach)
        elif args.experiment == 'topk_analysis':
            trainer.analyze_topk(testing_graph_set=args.testset,k_range=args.k_range,top_k_approach=args.topk_approach)
        elif args.experiment == 'diversity_analysis':
            trainer.analyze_solution_diversity(testing_graph_set=args.testset,top_k_approach=args.topk_approach,test_k=args.test_k)
    trainer.close()
           

if __name__ == "__main__":
    main()
