from argparse import ArgumentParser
import time
from pprint import pprint
from pycocoevalcap.eval import Bleu, Cider, Meteor, PTBTokenizer, Rouge, Spice


def read_plaintext_file(file):
    """
    Plaintext, TSV file.

    This is a caption   This is another caption
    This is yet another caption
    """

    data = {}
    with open(file) as fin:
        for i, line in enumerate(fin):
            captions = line.strip().split("\t")
            data[i] = [{"caption": c} for c in captions]
    return data


def compute_nlg_metrics(predictions, gold_standard):
    tokenizer = PTBTokenizer()

    predictions = tokenizer.tokenize(predictions)
    ground_truth = tokenizer.tokenize(gold_standard)

    scorers = [
        # (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        # (Meteor(), "METEOR"),
        # (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
        # (Spice(), "SPICE"),
    ]

    summary = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ground_truth, predictions)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                summary[m] = sc
        else:
            summary[method] = score
    print()
    pprint(summary)
    return summary


def get_cli_args():
    parser = ArgumentParser()
    # parser.add_argument("prediction_file", help="File containing predictions.")
    # parser.add_argument("gold_file", help="File containing gold captions.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_evaluation():
    args = get_cli_args()
    p = "pretrained_coco_no_ft_tested_coco.txt"
    g = "gt_coco_karpathy5k_split.txt"

    predictions = read_plaintext_file(p)
    gold_standard = read_plaintext_file(g)
    # predictions = {k: v for k, v in predictions.items() if k < 16}
    # gold_standard = {k: v for k, v in gold_standard.items() if k < 16}
    predictions = {0: [{'caption': 'A woman standing on a snowy slope with skis.'}], 1: [{'caption': 'A clock with a bird on it is on a table.'}], 2: [{'caption': 'A man standing in front of a grill with a hot dog.'}], 3: [{'caption': 'A baseball player swinging a bat at a ball.'}], 4: [{'caption': 'A living room with a chair, a table, and a chair.'}], 5: [{'caption': 'A group of cows grazing on a hillside.'}], 6: [{'caption': 'A man with a hat and a hat on a skateboard.'}], 7: [{'caption': 'A boat is in the water with a large fog.'}], 8: [{'caption': 'A glass case with a variety of glass vases.'}], 9: [{'caption': 'A baby is holding a teddy bear.'}], 10: [{'caption': 'A kitchen with a wall and a stove top.'}], 11: [{'caption': 'A boat is sitting on the water near a dock.'}], 12: [{'caption': 'A tour bus with a large poster on it.'}], 13: [{'caption': 'A bathroom with a white toilet paper and a white towel.'}], 14: [{'caption': 'A bag of electronics, a bottle of water, a bag of cigarettes, a bag of cigarettes, a bag of cigarettes, a bag of cigarettes, a bag of cigarettes, a bag of cigarettes,'}], 15: [{'caption': 'A group of people standing in a park with umbrellas.'}]}
    gold_standard = {0: [{'caption': 'A man riding skis while holding two ski poles.'}, {'caption': 'an image of a man on skiis on slopes'}, {'caption': 'A person standing in the snow wearing skis.'}, {'caption': 'a person riding skis on a snowy surface'}, {'caption': 'A man smiles while wearing snow skis in a snowy yard.'}], 1: [{'caption': 'A small white clock mounted on top of a glass painting.'}, {'caption': 'A white clock decorated with grains and garlic.'}, {'caption': 'A small clock is decorated with baskets and barrels of food.'}, {'caption': 'Colorful clock designed with various different food stuffs.'}, {'caption': 'A decorative white clock sits on a white surface.'}], 2: [{'caption': 'A man standing in front of a BBQ grill.'}, {'caption': 'A person cooking food on a grill in the middle of the woods.'}, {'caption': 'A man is standing in front of a grill with an umbrella.'}, {'caption': 'Man grilling on barbecue in backyard holding an umbrella.'}, {'caption': 'a guy with a blue umbrella next to a grill'}], 3: [{'caption': 'A baseball player is trying to hit a ball.'}, {'caption': 'A baseball player swinging his bat at a baseball.'}, {'caption': 'A baseball player that just hit a baseball.'}, {'caption': 'A batter is swinging at the ball at home plate.'}, {'caption': 'A baseball player has just hit a ball.'}], 4: [{'caption': 'The furniture is posed in the room with a sign that says do not touch.'}, {'caption': 'The room is crowded with many things including chairs, a bicycle, and a table with cups on it.'}, {'caption': 'A group of chairs sitting around a table.'}, {'caption': 'A living area with a number of chairs'}, {'caption': 'there is a small table with tea cups and three chairs around it'}], 5: [{'caption': 'A black cow standing next to a brown cow on a lush green hillside.'}, {'caption': 'Two cows at the edge of a large hill.'}, {'caption': 'A black steer standing next to a brown steer lying down.'}, {'caption': 'two cows on the edge of a cliff with mountains in the background'}, {'caption': 'Two cows on a hill above a valley and mountains on the other side.'}], 6: [{'caption': 'A young man holding a milk jug and standing on a skateboard.'}, {'caption': 'a person riding a skate board at a skate park'}, {'caption': 'A man on a skateboard holding a gallon of tea.'}, {'caption': 'A picture of a person on a skateboard.'}, {'caption': 'The man is standing on a skate board.'}], 7: [{'caption': 'Boat on the water under low clouds near land.'}, {'caption': 'a boat out on water with a city in the background'}, {'caption': 'A boat out to sea with mist coming off the water'}, {'caption': 'Boat sailing across a large body of water with a Myst cloud.'}, {'caption': 'A small boat in the water is hidden by the fog.'}], 8: [{'caption': 'The vases are displayed in the glass case.'}, {'caption': 'a number of different sized and colored vases behind a glass'}, {'caption': 'A glass case full of glass vases and candlesticks'}, {'caption': 'there are many glass vases in a stand'}, {'caption': 'A very nice looking glass case with many vases.'}], 9: [{'caption': 'a close up of a baby with a stuffed animal'}, {'caption': 'A little baby holding a teddy bear with both hands.'}, {'caption': 'baby in a stripped shirt holding a teddy bear'}, {'caption': 'A baby is sitting and holding a teddy bear.'}, {'caption': 'A baby that is sitting and hugging a teddy bear.'}], 10: [{'caption': 'a ratchet ass room with a shelf on the wall'}, {'caption': 'A kitchen scene with a dual element and boxes.'}, {'caption': 'A poor looking room has a hot plate kind of stove.'}, {'caption': 'There is an empty and dirty storage room.'}, {'caption': 'In the corner of this room there are several things including a shelf and a plug in.'}], 11: [{'caption': 'This black and white photo was taken by water.'}, {'caption': 'Boat launch ramp with some boats on the banks nearby.'}, {'caption': 'A river with not to much water lined with boats.'}, {'caption': 'A black and white image of many boats outside of the lake.'}, {'caption': 'A group of boats sit on the side of the water.'}], 12: [{'caption': 'A tour bus with disney land painted on the side'}, {'caption': 'A passenger bus designed with images of Disneyland.'}, {'caption': 'There is a bus with pictures on the side of it.'}, {'caption': 'A tour bus displaying a Disneyland scene advertisement.'}, {'caption': 'A large bus with advertisements on the side of it'}], 13: [{'caption': 'A bathroom with a shower curtain that looks like an error screen in a Microsoft windows program.'}, {'caption': 'A bathroom that has a shower curtain to look like a webpage.'}, {'caption': 'A bathroom with a computer screen printed shower curtain.'}, {'caption': 'A strange looking shower curtain in an ordinary looking bathroom.'}, {'caption': 'A bathroom with a shower curtain printed with a browser error message.'}], 14: [{'caption': "A purse sitting next to it's contents on top of a table."}, {'caption': 'The contents of a bag spread out on a couch'}, {'caption': 'Supplies scattered out on a couch cushion next to a purse.'}, {'caption': 'an image of a variety of electronics and personables'}, {'caption': 'Many items have been arranged neatly near a travel bag.'}], 15: [{'caption': "A group of people standing in the rain under some umbrella's."}, {'caption': 'People with umbrellas looking towards the grassy area'}, {'caption': 'people walking in the park in the rain\n'}, {'caption': 'A large group of people standing with some umbrellas.'}, {'caption': 'Some people are standing with umbrellas in the rain.'}]}
    start = time.time()
    compute_nlg_metrics(predictions=predictions, gold_standard=gold_standard)

    print(f"Total time  elapsed : {time.time() - start}")

if __name__ == "__main__":
    run_evaluation()