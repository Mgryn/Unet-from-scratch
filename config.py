import argparse

arg_list = []
parser = argparse.ArgumentParser(description="Unet")

parser.add_argument('--data_dir', type=str, default = 'data/', help='data directory')

parser.add_argument('--seed', type=int, default = 42, help='random seed for training')

parser.add_argument('--epochs', type=int, default = 50, help='Number of training epochs')

parser.add_argument('--batch_size', type=int, default= 1)

parser.add_argument('--lr', type=float, default = 0.99, help='Learning rate')

parser.add_argument('--mn', type = float, default = 0.99, help='Momentum value')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
