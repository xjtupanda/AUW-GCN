import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for XXX')
    
    parser.add_argument("--dataset", type=str, default='cas(me)^2',
                        choices=['cas(me)^2', 'samm'])
    parser.add_argument("--output", type=str, default='./output/casme',
                        help='dir for saving logs, models, etc.')
    parser.add_argument("--subject", type=str, default='casme_015',
                        help='Leave out subject for evaluation')
    
    args = parser.parse_args()
    return args