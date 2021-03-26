import argparse
import json
from pathlib import Path

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
    export_results_to_csv
)
from reporter.preprocessing.dataset import create_dataset
from reporter.util.config import Config
from reporter.util.constant import Phase
from reporter.util.logging import create_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='reporter.predict')
    parser.add_argument('--device',
                        type=str,
                        metavar='DEVICE',
                        default='cpu',
                        help='`cuda:n` where `n` is an integer, or `cpu`')
    parser.add_argument('--debug',
                        dest='is_debug',
                        action='store_true',
                        default=False)
    parser.add_argument('--config',
                        type=str,
                        dest='dest_config',
                        metavar='FILENAME',
                        default='config.toml',
                        help='specify config file (default: `config.toml`)')
    parser.add_argument('--model_dir',
                        type=str,
                        dest='model_dir',
                        default='output')
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(args.dest_config)
    device = torch.device(args.device)

    model_dir = Path(args.model_dir)
    dest_dir = model_dir / Path('add-eval')

    dest_log = dest_dir / Path('reporter-eval.log')

    logger = create_logger(dest_log, is_debug=args.is_debug)
    config.write_log(logger)

    # === Dataset ===
    (token_field, train, valid, test) = create_dataset(config, device)
    vocab_size = len(token_field.vocab)
    vocab_path = model_dir / Path('reporter.vocab')
    with vocab_path.open(mode='rb') as f:
        token_field.vocab = torch.load(f)

    # === Model ===
    encoder = Encoder(config, device)
    decoder = Decoder(config, vocab_size, device)
    model = EncoderDecoder(encoder, decoder, device)

    model_path = model_dir / Path('reporter.model')
    with model_path.open(mode='rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))

    optimizer = None
    criterion = torch.nn.NLLLoss(reduction='none',
                                 ignore_index=token_field.vocab.stoi[token_field.pad_token])
    neg_criterion = torch.nn.MultiMarginLoss(margin=config.tau_margin)
    try:
        with Path(config.neg_rule_file).open(mode='r') as f:
            neg_rules = json.load(f)
    except FileNotFoundError as e:
        neg_rules = None
        if config.neg_enabled:
            raise e
    negative = NegativeSampling(token_field=token_field, neg_enabled=config.neg_enabled, neg_rules=neg_rules,
                                neg_rule_tag_classes=config.neg_rule_tag_classes,
                                neg_samples_ratio=config.neg_samples_ratio,
                                neg_loss_type=config.neg_loss_type,
                                neg_loss_ratio=config.neg_loss_ratio)

    model.eval()
    # === Valid ===
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

    valid_log = ['Valid Loss: {:.2f}'.format(valid_result.loss), 'Valid BLEU: {:.10f}'.format(valid_bleu)]

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

    test_log = ['Test Loss: {:.2f}'.format(test_result.loss), 'Test BLEU: {:.10f}'.format(test_bleu)]

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
