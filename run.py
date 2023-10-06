import argparse
import os

# This script is to run only one

def parse_args(argv=None):
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('-m', default='MH', type=str,
                        help='which feature process module to select. (e.g., STD, OE, AD, MH)')
    parser.add_argument('-b', default='resnet', type=str,
                        help='which backbone to select. (eg., mlp, resnet)')
    parser.add_argument('-d', default='AT', type=str,
                        help='which data set to select. (eg., CA, AT, HE, HI, JA, YE)')

    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = parse_args()

    model_script = f'bin/{args.b}_{args.m}.py'
    config_script = f'configs/{args.d}/{args.b}+{args.m}.toml'
    output_dir = f'output/{args.d}'

    if os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    os.system(f'python {model_script} {config_script} -o {output_dir} -f')

