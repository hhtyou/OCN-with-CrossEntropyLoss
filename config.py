import argparse

def parse_args():
    parse = argparse.ArgumentParser("config stn args")
    parse.add_argument("--lr",default=0.0005,
    type=float,help="learning rate")
    parse.add_argument("--epoch_nums",default=20000,
    type=int,help="iterated epochs")
    parse.add_argument("--use_stn",default=True,
    type=bool,help="whether to use STN module")
    parse.add_argument("--RESUME",default=True,
    type=bool,help="whether to use pretrain module")
    parse.add_argument("--batch_size",default=80,
    type=int,help="batch size")
    parse.add_argument("--val_batch_size",default=16,
    type=int,help="val batch size")
    parse.add_argument("--test_batch_size",default=1,
    type=int,help="test batch size")
    parse.add_argument("--use_eval",default=True,
    type=bool,help="whether to evaluate")
    parse.add_argument("--use_visual",default=True,
    type=bool,help="visual STN transform image")
    parse.add_argument("--use_gpu",default=True,
    type=bool,help="whether to use GPU")
    parse.add_argument("--show_net_construct",default=False,
    type=bool,help="print net construct info")
    return parse.parse_args()


