import csv
from pathlib import Path
from typing import Dict, List

from reporter.core.train import RunResult
from reporter.util.constant import Phase


def export_results_to_csv(dest_dir: Path, result: RunResult, phase: Phase) -> None:

    header = ['article_id',
              'gold tokens (tag)',
              'gold tokens (num)',
              'pred tokens (tag)',
              'pred tokens (num)']
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_file = dest_dir / Path('reporter-%s.csv' % phase.value)

    with output_file.open(mode='w') as w:
        writer = csv.writer(w, delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        for (article_id, gold_sent, gold_sent_num, pred_sent, pred_sent_num) in \
                zip(result.article_ids,
                    result.gold_sents,
                    result.gold_sents_num,
                    result.pred_sents,
                    result.pred_sents_num):
            writer.writerow([article_id,
                             '|'.join(gold_sent),
                             '|'.join(gold_sent_num),
                             '|'.join(pred_sent),
                             '|'.join(pred_sent_num)])


def export_neg_info_tuples(dest_dir: Path, neg_info_tuples: List[tuple], epoch: int) -> None:
    header = ['epoch', 'article_id', 'rule_tags', 'idx', 'pos_token', 'neg_token']
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_file = dest_dir / Path('neg-info.csv')

    with output_file.open(mode='a') as w:
        writer = csv.writer(w, delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        for article_id, rule_tags, idx, pos_token, neg_token in neg_info_tuples:
            writer.writerow([epoch,
                             article_id,
                             rule_tags,
                             str(idx),
                             pos_token,
                             neg_token])


def export_neg_eval_to_csv(dest_dir: Path, neg_eval: Dict, phase: Phase) -> None:
    header = ['rule_tag_class',
              'rule_tag',
              'predP/goldP',
              'predN/goldP',
              'predPN/goldP',
              'pred-other/goldP',
              'gold-count',
              'goldP/predP',
              'goldN/predP',
              'goldPN/predP',
              'gold-other/predP',
              'pred-count',
              'Recall(predP/gold-count)',
              'Precision(goldP/pred-count)',
              'Error1(predN/gold-count)',
              'Error2(goldN/pred-count)']
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_file = dest_dir / Path('neg-eval-%s.csv' % phase.value)

    with output_file.open(mode='w') as w:
        writer = csv.writer(w, delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        for rule_tag_class in neg_eval.keys():
            cnt_dict = {'predP/goldP': 0, 'predN/goldP': 0, 'predPN/goldP': 0, 'pred-other/goldP': 0, 'gold-count': 0,
                        'goldP/predP': 0, 'goldN/predP': 0, 'goldPN/predP': 0, 'gold-other/predP': 0, 'pred-count': 0}

            for rule_tag in neg_eval[rule_tag_class].keys():
                if neg_eval[rule_tag_class][rule_tag]['gold-count']:
                    rec = float(neg_eval[rule_tag_class][rule_tag]['predP/goldP']) / \
                        neg_eval[rule_tag_class][rule_tag]['gold-count']
                    err1 = float(neg_eval[rule_tag_class][rule_tag]['predN/goldP']) / \
                        neg_eval[rule_tag_class][rule_tag]['gold-count']
                else:
                    rec = 0.0
                    err1 = 0.0
                if neg_eval[rule_tag_class][rule_tag]['pred-count']:
                    prec = float(neg_eval[rule_tag_class][rule_tag]['goldP/predP']) / \
                        neg_eval[rule_tag_class][rule_tag]['pred-count']
                    err2 = float(neg_eval[rule_tag_class][rule_tag]['goldN/predP']) / \
                        neg_eval[rule_tag_class][rule_tag]['pred-count']
                else:
                    prec = 0.0
                    err2 = 0.0
                writer.writerow([rule_tag_class,
                                 rule_tag,
                                 neg_eval[rule_tag_class][rule_tag]['predP/goldP'],
                                 neg_eval[rule_tag_class][rule_tag]['predN/goldP'],
                                 neg_eval[rule_tag_class][rule_tag]['predPN/goldP'],
                                 neg_eval[rule_tag_class][rule_tag]['pred-other/goldP'],
                                 neg_eval[rule_tag_class][rule_tag]['gold-count'],
                                 neg_eval[rule_tag_class][rule_tag]['goldP/predP'],
                                 neg_eval[rule_tag_class][rule_tag]['goldN/predP'],
                                 neg_eval[rule_tag_class][rule_tag]['goldPN/predP'],
                                 neg_eval[rule_tag_class][rule_tag]['gold-other/predP'],
                                 neg_eval[rule_tag_class][rule_tag]['pred-count'],
                                 rec,
                                 prec,
                                 err1,
                                 err2])
                for k in cnt_dict.keys():
                    cnt_dict[k] += neg_eval[rule_tag_class][rule_tag][k]
            if cnt_dict['gold-count']:
                rec = float(cnt_dict['predP/goldP']) / cnt_dict['gold-count']
                err1 = float(cnt_dict['predN/goldP']) / cnt_dict['gold-count']
            else:
                rec = 0.0
                err1 = 0.0
            if cnt_dict['pred-count']:
                prec = float(cnt_dict['goldP/predP']) / cnt_dict['pred-count']
                err2 = float(cnt_dict['goldN/predP']) / cnt_dict['pred-count']
            else:
                prec = 0.0
                err2 = 0.0
            writer.writerow([rule_tag_class,
                             'sum',
                             cnt_dict['predP/goldP'],
                             cnt_dict['predN/goldP'],
                             cnt_dict['predPN/goldP'],
                             cnt_dict['pred-other/goldP'],
                             cnt_dict['gold-count'],
                             cnt_dict['goldP/predP'],
                             cnt_dict['goldN/predP'],
                             cnt_dict['goldPN/predP'],
                             cnt_dict['gold-other/predP'],
                             cnt_dict['pred-count'],
                             rec,
                             prec,
                             err1,
                             err2])
