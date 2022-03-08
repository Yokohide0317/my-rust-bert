use tch::Device;
use rust_bert::resources::{LocalResource, Resource};
use rust_bert::resources::Resource::Local;
use std::path::PathBuf;
use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
use rust_bert::pipelines::question_answering::QuestionAnsweringConfig;

fn input(display: String) -> String {
    let mut text = String::new();
    println!("{}: ", display);
    std::io::stdin().read_line(&mut text).unwrap();
    return text.trim().to_string();
}

fn get_resource(item: String) -> Resource {
    let mut model_dir = PathBuf::from("/home/yokohide/workspace/rust/bert-large-uncased-whole-word-masking-finetuned-squad/");
    model_dir.push(&item);
    println!("{:?}", model_dir);
    let resource = Local(LocalResource{
        local_path: model_dir,
    });
    return resource;
}

fn main() -> anyhow::Result<()> {
    
    println!("Loading Files...");
    let model_resource = get_resource(String::from("rust_model.ot"));
    let vocab_resource = get_resource(String::from("vocab.txt"));
    let config_resource = get_resource(String::from("config.json"));
    println!("");

    let generate_config = QuestionAnsweringConfig {     
        model_type: rust_bert::pipelines::common::ModelType::Bert,
        model_resource,
        config_resource,
        vocab_resource,
        //merges_resource,

        // パラメーター調整
        device: Device::cuda_if_available(),
        lower_case: false,
        strip_accents: Some(false),
        doc_stride: 80,
        max_query_length: 80,
        max_seq_length: 800,
        max_answer_length: 20,
        ..Default::default()
    };

    let model = QuestionAnsweringModel::new(generate_config)?;
    //let question = String::from("Where does Amy live ?");
    //let context = String::from("Amy lives in Amsterdam");
    loop {
        let context = input(String::from("context"));
        if context=="QUIT" { break; }
        let question = input(String::from("question"));

        let answers = model.predict(&vec![QaInput { question, context }], 1, 32);

        println!("Answer: {:?}\n", answers[0][0].answer);
    }
    Ok(())
}
