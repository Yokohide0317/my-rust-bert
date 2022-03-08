import torch
from transformers import GPT2Model, GPT2Config
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    while True:
        input_context = input("Input: ")
        
        start = time.perf_counter()
        input_ids = tokenizer.encode(input_context, return_tensors='pt')

        output = model.generate(
            input_ids, 
            max_length=50, 
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            num_return_sequences=1,
            repetition_penalty=1.6,
        )

        print(tokenizer.decode(output[0], skip_special_tokens=True))
        print("{:.3}".format(time.perf_counter() - start))

if __name__ == "__main__":
    main()
