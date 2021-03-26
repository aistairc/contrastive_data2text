import argparse
import json
import random
import warnings
from datetime import datetime
from functools import reduce
from pathlib import Path

import numpy
import torch

from reporter.core.negative import NegativeSampling
from reporter.core.network import (
    Decoder,
    Encoder,
    EncoderDecoder
)
from reporter.core.train import run
from reporter.postprocessing.bleu import calc_bleu
from reporter.postprocessing.export import (
    export_neg_eval_to_csv,
    export_neg_info_tuples,
    export_results_to_csv
)
from reporter.preprocessing.dataset import create_dataset
from reporter.util.config import Config
from reporter.util.constant import Phase
from reporter.util.logging import create_logger


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(prog='reporter')
    parser.add_argument('--device',
                        type=str,
                        metavar='DEVICE',
                        default='cpu',
                        help='`cuda:n` where `n` is an integer, or `cpu`')
    parser.add_argument('--debug',
                        dest='is_debug',
                        action='store_true',
                        default=False,
                        help='show detailed messages while execution')
    parser.add_argument('--config',
                        type=str,
                        dest='dest_config',
                        metavar='FILENAME',
                        default='config.toml',
                        help='specify config file (default: `config.toml`)')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        metavar='FILENAME')
    parser.add_argument('-o',
                        '--output_subdir',
                        type=str,
                        metavar='DIRNAME')
    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if not args.is_debug:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)

    config = Config(args.dest_config)

    # seed
    seed = config.seed
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)

    now = datetime.today().strftime('reporter-%Y-%m-%d-%H-%M-%S')
    dest_dir = config.dir_output / Path(now) \
        if args.output_subdir is None \
        else config.dir_output / Path(args.output_subdir) / Path(now)

    dest_log = dest_dir / Path('reporter.log')

    logger = create_logger(dest_log, is_debug=args.is_debug)
    config.write_log(logger)

    message = 'start main (is_debug: {}, device: {})'.format(args.is_debug, args.device)
    logger.info(message)

    # === Alignment ===
    has_all_alignments = \
        reduce(lambda x, y: x and y,
               [(config.dir_output / Path('alignment-{}.json'.format(phase.value))).exists()
                for phase in list(Phase)])
    assert has_all_alignments

    # === Dataset ===
    (token_field, train, valid, test) = create_dataset(config, device)

    vocab_size = len(token_field.vocab)
    dest_vocab = dest_dir / Path('reporter.vocab')
    with dest_vocab.open(mode='wb') as f:
        torch.save(token_field.vocab, f)

    encoder = Encoder(config, device)
    decoder = Decoder(config, vocab_size, device)
    model = EncoderDecoder(encoder, decoder, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.NLLLoss(reduction='none',
                                 ignore_index=token_field.vocab.stoi[token_field.pad_token])
    neg_criterion = torch.nn.MultiMarginLoss(margin=config.tau_margin, reduction='none')
    try:
        with Path(config.neg_rule_file).open(mode='r') as f:
            neg_rules = json.load(f)
    except FileNotFoundError as e:
        neg_rules = None
        if config.neg_enabled:
            raise e
    negative = NegativeSampling(token_field=token_field,
                                neg_enabled=config.neg_enabled,
                                neg_loss_type=config.neg_loss_type,
                                neg_rules=neg_rules,
                                neg_rule_tag_classes=config.neg_rule_tag_classes,
                                neg_samples_ratio=config.neg_samples_ratio,
                                neg_loss_ratio=config.neg_loss_ratio)

    # === Train ===
    dest_model = dest_dir / Path('reporter.model')
    prev_valid_bleu = 0.0
    max_bleu = 0.0
    best_epoch = 0
    early_stop_counter = 0
    for epoch in range(config.n_epochs):
        logger.info('start epoch {}'.format(epoch))
        # train
        model.train()
        train_result = run(train,
                           token_field,
                           model,
                           optimizer,
                           criterion,
                           neg_criterion,
                           Phase.Train,
                           negative,
                           logger)
        train_bleu = calc_bleu(train_result.gold_sents, train_result.pred_sents)
        # valid
        model.eval()
        valid_result = run(valid,
                           token_field,
                           model,
                           optimizer,
                           criterion,
                           neg_criterion,
                           Phase.Valid,
                           negative,
                           logger)
        valid_bleu = calc_bleu(valid_result.gold_sents, valid_result.pred_sents)

        train_valid_log = ['epoch: {0:4d}'.format(epoch),
                           'training loss: {:.2f}'.format(train_result.loss),
                           'training lm-loss: {:.2f}'.format(train_result.lm_loss),
                           'training neg-loss: {:.6f}'.format(train_result.loss - train_result.lm_loss),
                           'training BLEU: {:.4f}'.format(train_bleu)]

        if train_result.prob_wins:
            num_probs = len(train_result.prob_wins)
            num_wins = sum(train_result.prob_wins)
            train_valid_log += \
                ['training Prob Wins: {:.10f} ({:d}/{:d})'.format(float(num_wins) / num_probs, num_wins, num_probs)]

        train_valid_log += ['validation loss: {:.2f}'.format(valid_result.loss),
                            'validation BLEU: {:.4f}'.format(valid_bleu)]

        if valid_result.prob_wins:
            num_probs = len(valid_result.prob_wins)
            num_wins = sum(valid_result.prob_wins)
            train_valid_log += \
                ['validation Prob Wins: {:.10f} ({:d}/{:d})'.format(float(num_wins) / num_probs, num_wins, num_probs)]

        s = ' | '.join(train_valid_log)
        logger.info(s)
        if negative.neg_enabled and negative.neg_loss_type == 'sentence':
            export_neg_info_tuples(dest_dir, train_result.neg_info_tuples, epoch)

        if max_bleu < valid_bleu:
            torch.save(model.state_dict(), str(dest_model))
            max_bleu = valid_bleu
            best_epoch = epoch

        early_stop_counter = early_stop_counter + 1 \
            if prev_valid_bleu > valid_bleu \
            else 0
        if early_stop_counter == config.patience:
            logger.info('EARLY STOPPING')
            break
        prev_valid_bleu = valid_bleu

    # === Valid ===
    with dest_model.open(mode='rb') as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    valid_result = run(valid,
                       token_field,
                       model,
                       optimizer,
                       criterion,
                       neg_criterion,
                       Phase.Valid,
                       negative,
                       logger,
                       gen_text=True)
    valid_bleu = calc_bleu(valid_result.gold_sents, valid_result.pred_sents)

    valid_log = ['epoch: {:04d}'.format(best_epoch),
                 'Valid Loss: {:.2f}'.format(valid_result.loss),
                 'Valid BLEU: {:.10f}'.format(valid_bleu)]
    if valid_result.prob_wins:
        num_probs = len(valid_result.prob_wins)
        num_wins = sum(valid_result.prob_wins)
        valid_log += ['Valid Prob Wins: {:.10f} ({:d}/{:d})'.format(float(num_wins) / num_probs, num_wins, num_probs)]

    export_results_to_csv(dest_dir, valid_result, Phase.Valid)

    neg_eval_valid = negative.evaluate(valid_result.gold_sents, valid_result.pred_sents)
    export_neg_eval_to_csv(dest_dir, neg_eval_valid, Phase.Valid)

    # === Test ===
    test_result = run(test,
                      token_field,
                      model,
                      optimizer,
                      criterion,
                      neg_criterion,
                      Phase.Test,
                      negative,
                      logger,
                      gen_text=True)
    test_bleu = calc_bleu(test_result.gold_sents, test_result.pred_sents)

    test_log = ['epoch: {:04d}'.format(best_epoch),
                'Test Loss: {:.2f}'.format(test_result.loss),
                'Test BLEU: {:.10f}'.format(test_bleu)]
    if test_result.prob_wins:
        num_probs = len(test_result.prob_wins)
        num_wins = sum(test_result.prob_wins)
        test_log += ['Test Prob Wins: {:.10f} ({:d}/{:d})'.format(float(num_wins) / num_probs, num_wins, num_probs)]

    export_results_to_csv(dest_dir, test_result, Phase.Test)

    neg_eval_test = negative.evaluate(test_result.gold_sents, test_result.pred_sents)
    export_neg_eval_to_csv(dest_dir, neg_eval_test, Phase.Test)

    s = ' | '.join(valid_log)
    logger.info(s)
    s = ' | '.join(test_log)
    logger.info(s)


if __name__ == '__main__':
    main()
