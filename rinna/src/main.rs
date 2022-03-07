use rust_bert::gpt2::GPT2Generator;
use rust_bert::pipelines::common::TokenizerOption;
use rust_bert::pipelines::generation_utils::{ GenerateConfig, LanguageGenerator };
use rust_bert::resources::LocalResource;
use rust_bert::resources::Resource::Local;
use std::path::PathBuf;
use tch::Device;

fn input() -> String {
    let mut text = String::new();
    println!("Input: ");
    std::io::stdin().read_line(&mut text).unwrap();
    return text.trim().to_string();
}

fn main() -> anyhow::Result<()> {
    let model_dir = PathBuf::from("/home/yokohide/workspace/rust/convert/gpt2/");

    let mut model_path = model_dir.clone();
    model_path.push("rust_model.ot");
    let model_resource = Local(LocalResource{ 
        local_path: model_path,
    });
     
    let mut vocab_path = model_dir.clone();
    vocab_path.push("vocab.json");
    let vocab_resource = Local(LocalResource{
        local_path: vocab_path,
    });

    let mut config_path = model_dir.clone();
    config_path.push("config.json");
    let config_resource = Local(LocalResource {
        local_path: config_path,
    });

    let merges_resource = vocab_resource.clone();   

    let generate_config = rust_bert::pipelines::text_generation::TextGenerationConfig {//GenerateConfig {        
        model_type: rust_bert::pipelines::common::ModelType::GPT2,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        device: Device::Cpu,
        repetition_penalty: 1.6,
        //min_length: 50,
        max_length: 30,
        do_sample: false,
        //early_stopping: true,
        num_beams: 1,
        temperature: 1.0,
        //top_k: 100,
        //top_p: 0.95,
        ..Default::default()
    };

//    let mut gpt2_model = GPT2Generator::new(generate_config)?; //tokenizer.into())?;
    let mut gpt2_model = rust_bert::pipelines::text_generation::TextGenerationModel::new(generate_config)?;
    gpt2_model.set_device(Device::cuda_if_available());
//    let tokenizer = gpt2_model.get_tokenizer();

//    let input_text = "I am a student."; //tokenizer.tokenize("I am a");
    let input_text = input();
    let first = input_text.clone();
    //println!("{:?}", input_text);
    let t1 = std::time::Instant::now();

    let output = gpt2_model.generate(&[input_text], None);
    for sentence in output {
        println!("{:?}", sentence);
    }
    //println!("{} {}", first, output[0].text);
    println!("Elapsed Time(ms):{}",t1.elapsed().as_millis()); 

    Ok(())
}

