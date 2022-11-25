import yaml  
import argparse

def _add_args_from_yaml(given_parser, input_args):
    given_parser.add_argument('-c','--config_yaml', default= None, type=str, metavar='FILE', help='YAML config file specifying default arguments')
    given_configs, remaining = given_parser.parse_known_args(input_args)
    if given_configs.config_yaml: 
        with open(given_configs.config_yaml, 'r', encoding='utf-8') as f: 
            cfgs = yaml.safe_load_all(f) 
            for cfg in cfgs:
                given_parser.set_defaults(**cfg)
        # temp_args, _ = given_parser.parse_known_args()
    else:
        print("no config file")

    return given_parser


# Priority: command arg > yaml
def argumentParse(input_args = None):
    print("start to argument parse")
    parser = argparse.ArgumentParser()
    # trainer config
    parser.add_argument('--batch_size', default=256, type=int, help='train batchsize') 
    parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--lr_f', '--flow_learning_rate', default=2e-5, type=float, help='initial flow learning rate')
    parser.add_argument('--noise_mode',  default='sym')
    parser.add_argument('--alpha_warmup', default=0.2, type=float, help='parameter for Beta (warmup)')
    parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
    parser.add_argument('--linear_u', default=340, type=float, help='weight for unsupervised loss')
    parser.add_argument('--lambda_u', default=30, type=float, help='weight for unsupervised loss')
    parser.add_argument('--lambda_flow_u', default=3, type=float, help='weight for unsupervised loss')
    parser.add_argument('--lamb_u_base', default=1, type=float, help='weight for unsupervised loss start value')
    parser.add_argument('--lambda_p', default=50, type=float, help='sharpening lamb')
    parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
    parser.add_argument('--Tu', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--Tx', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--num_epochs', default=350, type=int)
    parser.add_argument('-r', '--ratio', default=0.2 , type=float, help='noise ratio')
    parser.add_argument('--d_u',  default=0.7, type=float)
    parser.add_argument('--tau', default=5, type=float, help='filtering coefficient')
    parser.add_argument('--d_up',  default=0, type=float)
    parser.add_argument('--seed', default=123)
    parser.add_argument('--gpuid', default="0", help='comma separated list of GPU(s) to use.')
    parser.add_argument('--resume', action='store_true', help = 'Resume from the warmup checkpoint')
    parser.add_argument('--num_class', default=10, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--data_path', default='./data/cifar10', type=str, help='path to dataset')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--flow_modules', default="8-8-8-8", type=str)
    parser.add_argument('--name', default="", type=str)
    parser.add_argument('--fix', default='none', choices=['none', 'net', 'flow'], type=str)
    parser.add_argument('--pretrain', default='', type=str)
    parser.add_argument('--pseudo_std', default=0, type=float)
    parser.add_argument('--decay', default=0.9, type=float, help='Exponential Moving Average decay')
    parser.add_argument('--warm_up', default=10, type=int)
    parser.add_argument('--num_samples', default=50000, type=int)
    parser.add_argument('--ema_jsd', action='store_true', help = 'JSD Moving Average')
    parser.add_argument('--jsd_decay', default=0.9, type=float, help='Exponential Moving Average decay')
    parser.add_argument('--clip_grad', default=True, help = 'cliping grad')
    parser.add_argument('--pretrained', action='store_true', help = 'pretrained(Clothing1M)')
    parser.add_argument('--split', action='store_true', help = 'split flow loss to x, u')
    parser.add_argument('--cond_size', default=128, type=int)
    parser.add_argument('--isRealTask', default=False, type=bool, help='')
    parser.add_argument('--useUncertainty', default=False, type=bool, help='')
    parser.add_argument('--lambda_f', default=1.0, type=float, help='flow nll loss weight')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='SGD weight decay')
    parser.add_argument('--flow_sp', default=True, type=bool, help='flow sharpening')
    parser.add_argument('--flow_ssl', default=False, type=bool, help='flow ssl loss')
    parser.add_argument('--centering', default=False, type=bool, help='use centering')
    parser.add_argument('--w_ce', default=False, help = 'train with cross entrioy')
    
    # load yaml
    _add_args_from_yaml(parser, input_args)

    args = parser.parse_args(input_args)

    return args