import argparse
from config_utils import load_yaml
from DefenseTransformer import *

def main(args):
    if args.task:
        if args.task == "train":
            plot_path = "./data/trained_loss_with_mask_eighty.png"
            train(model = model, DL = DL, save_path= save_path, plot_path="./data/trained_loss_with_mask_eighty.png")
        elif args.task == "test":
            model.load_state_dict(torch.load(save_path))
            #test(model = model, DL = DL)

if __name__ == "__main__":

    print("Reading Config...")
    parser = argparse.ArgumentParser(description='Defense Transformer Model')
    args = parser.parse_args()
    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)

    main(args)