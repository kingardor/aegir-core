import importlib
import os
import sys
import tensorrt as trt
from loguru import logger
import argparse
import torch
from torch2trt import torch2trt

def argparser():
    parser = argparse.ArgumentParser("YOLOX TensorRT Engine Creator")
    parser.add_argument("-c", "--checkpoint", default=None, type=str, help='trained model path')
    parser.add_argument("-w", '--workspace', type=int, default=32, help='max workspace size in detect')
    parser.add_argument("-b", '--batch', type=int, default=1, help='max batch size in detect')
    return parser

@logger.catch
def main():
    args = argparser().parse_args()
    exp_file = 'yolox_s_debris.py'
    current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
    exp = current_exp.Exp()
    model = exp.get_model()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.cuda()
    logger.info("Model load on GPU sucessful.")

    model.head.decode_in_inference = False
    x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=True,
        log_level=trt.Logger.INFO,
        max_workspace_size=(1 << args.workspace),
        max_batch_size=args.batch,
    )
    torch.save(model_trt.state_dict(), 'model_' + args.checkpoint)

    engine_file = 'model_' + args.checkpoint[:-3] + 'engine'
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())

    logger.info("TensorRT model conversion done.")

if __name__ == '__main__':
    main()