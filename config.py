import argparse

arg_list = []
parser = argparse.ArgumentParser(description="Unet")

parser.add_argument('--data_dir', type=str, default='data/', help='data directory')

parser.add_argument('--seed', type=int, default=42, help='random seed for training')

parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')

parser.add_argument('--batch_size', type=int, default=1)

parser.add_argument('--valid_size', type=float, default=0.1, help='size of validation dataset size')

parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')

parser.add_argument('--mn', type=float, default=0.99, help='Momentum value')

parser.add_argument('--preprocessing', type=bool, default=False, help='if true, new augmented data will be generated')

parser.add_argument('--augmentations', type=int, default=20, help='number of augmentations performed on each images')

parser.add_argument('--train', type=bool, default=False, help='if true, the network will be trained, otherwise only evaluated')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
