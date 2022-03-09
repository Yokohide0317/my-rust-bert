use std::path::PathBuf;
use rust_bert::resources::LocalResource;
use rust_bert::resources::Resource::Local;
use rust_bert::pipelines::text_generation::{ TextGenerationConfig, TextGenerationModel };
use rust_bert::resources::Resource;
//use rust_tokenizers::tokenizer::Gpt2Tokenizer;
use rust_tokenizers::tokenizer::{SentencePieceTokenizer, Tokenizer};

fn input() -> String {
    let mut text = String::new();
    println!("Input: ");
    std::io::stdin().read_line(&mut text).unwrap();
    return text.trim().to_string();
}

fn get_resource(item: String) -> Resource {
    let mut model_dir = PathBuf::from("/home/yokohide/workspace/rust/japanese-gpt2-small/");
    model_dir.push(&item);
    println!("{:?}", model_dir);
    let resource = Local(LocalResource{
        local_path: model_dir,
    });
    return resource;
}

fn main() -> anyhow::Result<()> {
    /*
    let model_resource = get_resource(String::from("rust_model.ot"));
    let vocab_resource = get_resource(String::from("spiece.model")); 
    let config_resource = get_resource(String::from("config.json"));
    */
    let lower_case = false;
    let tokenizer = SentencePieceTokenizer::from_file("/home/yokohide/workspace/rust/japanese-gpt2-small/spiece.model", lower_case).unwrap();
    //let merges_resource = get_resource(String::from("merges.txt"));

    // configの作成
    let generate_config = TextGenerationConfig {     
        model_type: rust_bert::pipelines::common::ModelType::GPT2,
        model_resource,
        config_resource,
        vocab_resource,
        //merges_resource,

        // パラメーター調整
        repetition_penalty: 1.6,
        max_length: 30,
        do_sample: false,
        num_beams: 1,
        temperature: 1.0,
        ..Default::default()
    };
    
    // 上のconfigからモデル作成
    let model = TextGenerationModel::new(generate_config)?;
    //model.set_device(Device::cuda_if_available());
    loop {
        let input_text = input();
        // QUITで終了できるように
        if input_text == "QUIT" { break; }
        // 時間測定スタート
        let start = std::time::Instant::now();
        println!("Generating...");
        // 推論
        let output = model.generate(&[input_text], None);

        for sentence in output {
            println!("「{:?}」", sentence);
        }
        // 時間測定。差分を取る
        let stop = std::time::Instant::now();
        println!("<Time: {:.3}s>", (stop.duration_since(start).as_millis() as f64) / 1000.0);
        println!("\n");
    }
    Ok(())
}
