import argparse


def restore_quotationmark(src, pred):
    '''
    将误纠的双引号还原
    @param src: 原文
    @param pred: 预测文本
    @return: 复原双引号的预测文本
    '''
    assert len(src) == len(pred), (len(src), len(pred), src, pred)
    src = list(src)
    pred = list(pred)
    for i in range(len(src)):
        if src[i] == "“":
            pred[i] = '“'
        if src[i] == '”':
            pred[i] = '”'
    return ''.join(pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--para', type=str, required=True)
    parser.add_argument('--pred', type=str, required=True)
    args = parser.parse_args()

    para = [i for i in open(args.para)]
    src = [i.split('\t')[0] for i in para]
    pred = [i.strip() for i in open(args.pred)]

    restored_pred = [restore_quotationmark(s, p) for s, p in zip(src, pred)]
    with open(args.pred + ".postprocessed", 'w') as f:
        f.write('\n'.join(restored_pred))