import logging
import math
import random
import time
from dataclasses import dataclass
from datetime import timedelta

import tomllib
import torch
from tqdm import tqdm

from translation.manager import Batch, Manager

Criterion = torch.nn.CrossEntropyLoss
Optimizer = torch.optim.Optimizer
Scaler = torch.cuda.amp.GradScaler
Logger = logging.Logger


@dataclass
class DataArgs:
    train_data: str
    val_data: str
    lem_train: str
    lem_val: str


def train_epoch(
    batches: list[Batch],
    manager: Manager,
    criterion: Criterion,
    optimizer: Optimizer | None = None,
    scaler: Scaler | None = None,
) -> tuple[float, list[list[list[float]]]]:
    batch_conf = []
    total_loss, num_tokens = 0.0, 0
    for batch in tqdm(batches):
        src_nums, src_mask = batch.src_nums, batch.src_mask
        tgt_nums, tgt_mask = batch.tgt_nums, batch.tgt_mask
        batch_length = batch.length()

        if manager.dpe_embed:
            dict_mask, dict_data = None, batch._dict_data
        else:
            dict_mask, dict_data = batch.dict_mask, None

        with torch.amp.autocast(batch.device, enabled=False):
            logits, src_embs = manager.model(
                src_nums, tgt_nums, src_mask, tgt_mask, dict_mask, dict_data
            )
            loss = criterion(torch.flatten(logits[:, :-1], 0, 1), torch.flatten(tgt_nums[:, 1:]))
            if manager.dict and not manager.freq:
                probs, indices = logits[:, :-1].softmax(dim=-1), tgt_nums[:, 1:].unsqueeze(-1)
                tgt_prob = torch.gather(probs, dim=-1, index=indices).squeeze(-1).sum(dim=-1)
                grads = torch.autograd.grad(tgt_prob.unbind(), src_embs, retain_graph=True)
                batch_conf.append(torch.cat(grads).norm(p=1, dim=-1).tolist())
                del grads, probs, indices, tgt_prob

        if optimizer and scaler:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(manager.model.parameters(), manager.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        elif optimizer:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(manager.model.parameters(), manager.clip_grad)
            optimizer.step()

        total_loss += batch_length * loss.item()
        num_tokens += batch_length
        del logits, loss

    return total_loss / num_tokens, batch_conf


def train_model(data_args: DataArgs, manager: Manager, logger: Logger):
    model, vocab = manager.model, manager.vocab
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=vocab.PAD, label_smoothing=manager.label_smoothing
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=manager.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=manager.decay_factor, patience=manager.patience
    )
    # scaler = torch.cuda.amp.GradScaler()

    if manager.dict and not manager.freq:
        train_data, train_indices = manager.load_data(data_args.train_data)
        val_data, val_indices = manager.load_data(data_args.val_data)
    else:
        train_data, _ = manager.load_data(data_args.train_data, data_args.lem_train)
        val_data, _ = manager.load_data(data_args.val_data, data_args.lem_val)

    train_conf: list[list[list[float]]] = []
    val_conf: list[list[list[float]]] = []
    conf_lists: list[list[float]] = []
    shuffled_indices: list[int] = []

    epoch = patience = 0
    best_loss = torch.inf
    while epoch < manager.max_epochs:
        if manager.dict and not manager.freq:
            if epoch > 0:
                train_data = [x for _, x in sorted(zip(shuffled_indices, train_data))]
            if epoch > 1:
                train_conf = [x for _, x in sorted(zip(shuffled_indices, train_conf))]
                conf_lists = [sublist for outer in train_conf for sublist in outer]
                conf_lists = [x for _, x in sorted(zip(train_indices, conf_lists))]
                train_data, train_indices = manager.load_data(
                    data_args.train_data, data_args.lem_train, conf_lists
                )

                conf_lists = [sublist for outer in val_conf for sublist in outer]
                conf_lists = [x for _, x in sorted(zip(val_indices, conf_lists))]
                val_data, val_indices = manager.load_data(
                    data_args.val_data, data_args.lem_val, conf_lists
                )

            shuffled_indices = list(range(len(train_data)))
            combined_data = list(zip(train_data, shuffled_indices))
            random.shuffle(combined_data)
            train_data, shuffled_indices = list(zip(*combined_data))  # type: ignore[assignment]
        else:
            random.shuffle(train_data)

        model.train()
        start = time.perf_counter()
        train_loss, train_conf = train_epoch(train_data, manager, criterion, optimizer)
        elapsed = timedelta(seconds=(time.perf_counter() - start))

        model.eval()
        if manager.dict and not manager.freq:
            val_loss, val_conf = train_epoch(val_data, manager, criterion)
        else:
            with torch.no_grad():
                val_loss, _ = train_epoch(val_data, manager, criterion)
        scheduler.step(val_loss)

        checkpoint = f'[{str(epoch + 1).rjust(len(str(manager.max_epochs)), "0")}]'
        checkpoint += f' Training PPL = {math.exp(train_loss):.16f}'
        checkpoint += f' | Validation PPL = {math.exp(val_loss):.16f}'
        checkpoint += f' | Learning Rate = {optimizer.param_groups[0]["lr"]:.16f}'
        if manager.dict:
            checkpoint += f' | Dict Coverage = {(manager.dict_coverage * 100):.2f}'
        checkpoint += f' | Elapsed Time = {elapsed}'
        logger.info(checkpoint)
        print()

        if val_loss < best_loss:
            manager.save_model((epoch, val_loss), optimizer, scheduler)
            patience, best_loss = 0, val_loss
        else:
            patience += 1

        if optimizer.param_groups[0]['lr'] < manager.min_lr:
            logger.info('Reached Minimum Learning Rate.')
            break
        if patience >= manager.max_patience:
            logger.info('Reached Maximum Patience.')
            break
        epoch += 1
    else:
        logger.info('Maximum Number of Epochs Reached.')


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lang-pair', required=True, help='source-target language pair')
    parser.add_argument(
        '--train-data', metavar='FILE_PATH', required=True, help='parallel training data'
    )
    parser.add_argument(
        '--val-data', metavar='FILE_PATH', required=True, help='parallel validation data'
    )
    parser.add_argument('--lem-train', metavar='FILE_PATH', help='lemmatized training data')
    parser.add_argument('--lem-val', metavar='FILE_PATH', help='lemmatized validation data')
    parser.add_argument('--dict', metavar='FILE_PATH', help='bilingual dictionary')
    parser.add_argument('--freq', metavar='FILE_PATH', help='frequency statistics')
    parser.add_argument('--sw-vocab', metavar='FILE_PATH', required=True, help='subword vocab')
    parser.add_argument('--sw-model', metavar='FILE_PATH', required=True, help='subword model')
    parser.add_argument('--model', metavar='FILE_PATH', required=True, help='translation model')
    parser.add_argument('--log', metavar='FILE_PATH', required=True, help='logger output')
    parser.add_argument('--seed', type=int, help='random seed')
    args, unknown = parser.parse_known_args()

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    src_lang, tgt_lang = args.lang_pair.split('-')
    with open('translation/config.toml', 'rb') as config_file:
        config = tomllib.load(config_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i, arg in enumerate(unknown):
        if arg[:2] == '--' and len(unknown) > i:
            option, value = arg[2:].replace('-', '_'), unknown[i + 1]
            try:
                config[option] = (int if value.isdigit() else float)(value)
            except ValueError:
                config[option] = value

    manager = Manager(
        config,
        device,
        src_lang,
        tgt_lang,
        args.model,
        args.sw_vocab,
        args.sw_model,
        args.dict,
        args.freq,
    )
    data_args = DataArgs(args.train_data, args.val_data, args.lem_train, args.lem_val)

    if device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('high')

    logger = logging.getLogger('translation.logger')
    logger.addHandler(logging.FileHandler(args.log))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    train_model(data_args, manager, logger)


if __name__ == '__main__':
    main()
