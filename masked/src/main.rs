extern crate anyhow;

use rust_bert::bert::{BertConfig, BertForMaskedLM};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{BertTokenizer, MultiThreadedTokenizer, TruncationStrategy};
use rust_tokenizers::vocab::Vocab;
use tch::{nn, no_grad, Device, Tensor};

use std::path::PathBuf;

fn get_path(item: String) -> PathBuf {
    let mut resource_dir = PathBuf::from("/home/yokohide/workspace/rust/bert-base-japanese-whole-word-masking/");
    resource_dir.push(&item);
    println!("{:?}", resource_dir);
    return resource_dir;
}

fn input(display: String) -> String {
    let mut text = String::new();
    println!("{}: ", display);
    std::io::stdin().read_line(&mut text).unwrap();
    return text.trim().to_string();
}

fn main() -> anyhow::Result<()> {
    //    Resources paths
    let model_path: PathBuf = get_path(String::from("rust_model.ot"));
    let vocab_path: PathBuf = get_path(String::from("vocab.txt"));
    let config_path: PathBuf = get_path(String::from("config.json"));


    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer =
        BertTokenizer::from_file(vocab_path.to_str().unwrap(), false, false)?;
    let config = BertConfig::from_file(config_path);
    let bert_model = BertForMaskedLM::new(&vs.root(), &config);
    vs.load(model_path)?;

    //    Define input
    //let inp = String::from("明日は*に行きたい。");
    let inp = input(String::from("Input: "));
    let mut mask_index = 0;
    for (i, m) in inp.chars().enumerate() {
        if m == '*' {
            mask_index = i+1;
        }
    }
    let inp = inp.replace("*", "[MASK]");
    println!("{}", inp);

    let input = [inp,];
    
    let owakatied = &tokenizer.tokenize_list(&input)[0];
    let tokenized_input = tokenizer.encode_list(&owakatied, 128, &TruncationStrategy::LongestFirst, 0);

    let max_len = tokenized_input
        .iter()
        .map(|input| input.token_ids.len())
        .max()
        .unwrap();
    let tokenized_input = tokenized_input
        .iter()
        .map(|input| input.token_ids.clone())
        .map(|mut input| {
            input.extend(vec![0; max_len - input.len()]);
            input
        })
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let model_output = no_grad(|| {
        bert_model.forward_t(
            Some(&input_tensor),
            None,
            None,
            None,
            None,
            None,
            None,
            false,
        )
    });

    //    Print masked tokens
    let index_1 = model_output
        .prediction_scores
        .get(0)
        .get(mask_index.try_into().unwrap())
        .argmax(0, false);

    let word_1 = tokenizer.vocab().id_to_token(&index_1.int64_value(&[]));

    println!("{}", word_1); // Outputs "person" : "Looks like one [person] is missing"

    Ok(())
}
