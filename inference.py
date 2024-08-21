import csv
from whoosh.index import open_dir
from sklearn.preprocessing import normalize
from argparse import ArgumentParser
from DPR.DPR_Retriever import DPR_Retriever
import os.path as path
import json
import sys
import os


# Path to the directory containing DPR and subsequently DPR_module
base_path = os.path.join(os.path.dirname(__file__), 'DPR')
print(os.getcwd())

# Add both DPR and DPR_module directories to the PYTHONPATH
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, 'DPR_module'))

print(sys.path)

csv.field_size_limit(sys.maxsize)


class Inference:
    def __init__(self, tsv_file_path):
        self.dpr = DPR_Retriever()
        self.paragraphs = self.load_paragraphs(tsv_file_path)

    def load_paragraphs(self, file_path):
        """Load paragraphs from a TSV file."""
        paragraphs = {}
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                paragraphs[row['id']] = {
                    'text': row['text'], 'title': row['title']}
        return paragraphs

    def __normalize(self, results: dict):
        scores = [list(results.values())]
        scores = normalize(scores)
        for i, id in enumerate(results.keys()):
            results[id] = scores[0][i]

    def get_docs(self, question):
        # Retrieve documents using DPR
        dpr_result = self.dpr.get_top_docs_dpr(question, 100)

        if len(dpr_result) > 0:
            self.__normalize(dpr_result)

        # Match IDs and extract context
        final_result = {}
        for id, score in dpr_result.items():
            paragraph_data = self.paragraphs.get(id, {})
            final_result[id] = {
                'paragraph_id': id,
                'context': paragraph_data.get('text', 'Context not found.'),
                'title': paragraph_data.get('title', 'Title not found.'),
                'score': score
            }

        # Sort the results based on scores
        final_result = {k: v for k, v in sorted(
            final_result.items(), key=lambda item: item[1]['score'], reverse=True)}

        return final_result


def main():
    """
    parser = ArgumentParser()
    parser.add_argument('--question', type=str, required=True)
    args = parser.parse_args()

    question = args.question
    """
    # pip install transformers==4.41.1
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_id = "CohereForAI/aya-23-8B"
    print('Creating tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print('Creating model...')
    quantization_config = None
    QUANTIZE_4BIT = True
    if QUANTIZE_4BIT:
        import torch
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Format message with the command-r-plus chat template
    messages = [
        {"role": "user", "content": "Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz"}]
    print(messages)
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    # <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Anneme onu ne kadar sevdiğimi anlatan bir mektup yaz<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.3,
    )

    gen_text = tokenizer.decode(gen_tokens[0])
    print(gen_text)

    return
    tsv_file_path = './DPR/wiki/wikiAr.tsv'

    inference = Inference(tsv_file_path)
    # while True:
    question = "محمد حسني مبارك"  # input('Enter a question:')
    final_result = inference.get_docs(question)
    with open('result.json', mode='w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
