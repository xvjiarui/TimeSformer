import argparse

import torch

def convert(src, dst):
    # convert to pytorch style
    src_dict = torch.load(src, map_location='cpu')
    src_state_dict = src_dict['model']
    # save checkpoint
    checkpoint = dict()
    checkpoint['model'] = src_state_dict
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
