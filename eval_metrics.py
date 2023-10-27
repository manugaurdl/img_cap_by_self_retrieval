
import evaluate

def get_metric(metric,preds,refs):
    return evaluate.load(metric).compute(predictions=preds, references= refs)

def get_results(metric,preds,refs):
    print(metric.compute(predictions=preds, references= refs))

def main(preds, refs):

    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    print(get_results(bleu, preds,refs))
    print(get_results(meteor, preds,refs))


# if __name__ == '__main__':
#     preds = ["hey general kenobi","a cat is sitting on the tree"]
#     refs = [ ["hello there general kenobi"],["There is a cat sitting on the tree"]]
#     print(get_metric('bleu',preds, refs))
#     main(predictions, references)