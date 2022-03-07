extern crate anyhow;

use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::resources::Resource::Local;
use std::path::PathBuf;
use rust_bert::resources::LocalResource;
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
    model_path.push("rust_model.ot");model_path.push("rust_model.ot");
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

    //    Set-up model
    let generate_config = TextGenerationConfig {
        model_type: ModelType::GPT2,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        max_length: 30,
        do_sample: false,
        num_beams: 1,
        temperature: 1.0,
        num_return_sequences: 1,
        ..Default::default()
    };
    let mut model = TextGenerationModel::new(generate_config)?;
    model.set_device(Device::cuda_if_available());
    let input_context = String::from("My name");
    // let second_input_context = "The cat was";
    let output = model.generate(&[input_context], None);

    for sentence in output {
        println!("{:?}", sentence);
    }
    Ok(())
}
