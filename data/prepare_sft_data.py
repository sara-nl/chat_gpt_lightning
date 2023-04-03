import json
from datasets import load_dataset

def save(split, fp):
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        examples = []
        for data in dataset:
            examples.append(data["chosen"])
        json.dump(examples, fp)

def sft_set():
    """
    A simple script to create SFTDataset
    """
    with open("dataset_hhrlhf_train.json", "w") as fp:
        save("train", fp)
    with open("dataset_hhrlhf_test.json", "w") as fp:
        save("test", fp)
   
    sft_train = []
    with open("dataset_hhrlhf_train.json") as fp:
        hhtrain = json.load(fp)
        for h in hhtrain:
            sft_train.append(h)
            
    sft_test = []
    with open("dataset_hhrlhf_test.json") as fp:
        hhtest = json.load(fp)
        for h in hhtest:
            sft_test.append(h)

    with open("sft_train.json", "w") as fp:
        json.dump(sft_train, fp)
        print(len(sft_train))
        print(sft_train[-1])

    with open("sft_test.json", "w") as fp:
        json.dump(sft_test, fp)
        print(len(sft_test))
        print(sft_test[-1])

def main():
    sft_set()

if __name__ == "__main__":
    main()