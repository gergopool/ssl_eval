# Credits: https://github.com/facebookresearch/simsiam/blob/main/main_lincls.py
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torchvision import models

from typing import List, Union

import ssl_eval


def run_multi_gpu(model: torch.nn.Module,
                  dataset: str,
                  root: str,
                  n_views: int = 1,
                  batch_size: int = 4096,
                  seed: int = None,
                  method: Union[List[str], str] = "linear_eval",
                  devices: List[str] = ['0']):

    if seed:
        set_random_seed(seed)
    torch.multiprocessing.set_start_method("spawn")
    num_gpus = len(devices)
    port = random.randint(0, 9999) + 40000  # random port
    cudnn.benchmark = True

    if not isinstance(method, list):
        method = [method]

    for m in method:
        if m not in ["linear_eval", "knn", "snn", ""]:
            raise print(f"WARNING: Method {m} is unknown and therefore will be ignored.")

    result_dict = mp.Manager().dict() if len(devices) > 1 else {}
    common_args = (model, port, root, dataset, n_views, batch_size, devices, method, result_dict)
    if len(devices) > 1:
        mp.spawn(process, nprocs=num_gpus, args=(num_gpus, *common_args))
    else:
        process(0, 1, *common_args)

    return result_dict


def process(rank,
            world_size,
            model,
            port,
            data_root,
            dataset,
            n_views,
            batch_size,
            devices,
            method,
            results):

    init_distributed_setup(world_size, rank, devices, port)
    if world_size > 1:
        torch.distributed.barrier()

    torch.backends.cudnn.benchmark = True
    model.cuda()

    evaluator = ssl_eval.Evaluator(model,
                                   dataset,
                                   data_root,
                                   n_views=n_views,
                                   batch_size=batch_size)

    embs = evaluator.generate_embeddings()

    if "linear_eval" in method:
        lr = batch_size / 256 * 0.1
        acc = evaluator.linear_eval(embs, batch_size=batch_size, lr=lr)
        if rank == 0:
            results['linear_eval'] = acc

    if "knn" in method:
        accs = evaluator.knn(embs, k=[1, 5, 20])
        if rank == 0:
            results['knn'] = {}
            results['knn'][1] = accs[0]
            results['knn'][5] = accs[5]
            results['knn'][20] = accs[20]

    if "snn" in method:
        acc = evaluator.snn(embs)
        if rank == 0:
            results['snn'] = acc

    return embs, results


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    print('WARNING: You have chosen to seed training. '
          'This will turn on the CUDNN deterministic setting, '
          'which can slow down your training considerably! '
          'You may see unexpected behavior when restarting '
          'from checkpoints.')


def init_distributed_setup(world_size, rank, devices, port):
    if torch.cuda.is_available():
        device = torch.cuda.device(int(devices[rank]))
        torch.cuda.set_device(device)
    ssl_eval.distributed.init_distributed(port, rank_and_world_size=(rank, world_size))
    if rank == 0:
        print("Use GPU: {} for training".format(devices))
    return device


def get_model(state_dict, arch='resnet50'):
    model = models.__dict__[arch]()
    longest_module_name = ""
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
        if len(name) > len(longest_module_name):
            longest_module_name = name
    model.fc = torch.nn.Identity()

    # If any prefixes preceed keys, remove those
    state_dict, prefix = _find_item_and_prefix(state_dict, longest_module_name)
    if len(prefix):
        for k in list(state_dict.keys()):
            if k.startswith(prefix):
                state_dict[k[len(prefix):]] = state_dict[k]

    return model


def _find_item_and_prefix(d, lookup_key):

    queue = [d]
    found = False

    while queue and not found:
        current_dict = queue.pop(0)
        for key, child in current_dict.items():
            if key.endswith(lookup_key):
                state_dict = current_dict
                prefix = key[:len(lookup_key)]
                found = True
                break
            elif isinstance(child, dict):
                queue.append(child)

    if not found:
        raise ValueError(f"{lookup_key} not found anywhere with any prefixes in dictionary.")

    return state_dict, prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Eval Training')
    parser.add_argument('model_path', type=str)
    parser.add_argument('data_root', type=str)
    parser.add_argument('-d', '--dataset', type=str, default='imagenet')
    parser.add_argument('-a', '--arch', type=str, default='resnet50')
    parser.add_argument('-n', '--n-views', type=int, default=3)
    parser.add_argument('-b', '--batch-size', type=int, default=4096)
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('--devices', type=str, nargs='+', default=['0'])
    parser.add_argument('--method', type=str, nargs='+', default=['linear_eval', "knn", "snn"])
    args = parser.parse_args()

    state_dict = torch.load(args.model_path, map_location="cpu")
    model = get_model(state_dict, args.arch)

    run_multi_gpu(model,
                  args.dataset,
                  args.data_root,
                  args.n_views,
                  args.batch_size,
                  args.seed,
                  args.method,
                  args.devices)
