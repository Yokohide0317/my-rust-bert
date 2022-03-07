use rust_bert::gpt2::GPT2Generator;
use rust_bert::pipelines::generation_utils::GenerateConfig; //LanguageGenerator};
//use rust_bert::resources::RemoteResource; //  ResourceProvider};
use rust_bert::resources::{RemoteResource, Resource};
use tch::Device;
use rust_bert::pipelines::generation_utils::LanguageGenerator;

fn main() -> anyhow::Result<()> {
    let model_resource = Resource::Remote(RemoteResource::from_pretrained((
        "ja-bert",
        //"https://huggingface.co/jweb/japanese-soseki-gpt2-1b/resolve/main/rust_model.ot".into(),
        //"https://huggingface.co/Narsil/gpt2/blob/main/rust_model.ot",
        "https://huggingface.co/gpt2/blob/main/rust_model.ot",
    )));
    let config_resource = Resource::Remote(RemoteResource::new(
               //"https://huggingface.co/jweb/japanese-soseki-gpt2-1b/resolve/main/config.json".into(),
        //"https://huggingface.co/Narsil/gpt2/blob/main/config.json",
        "https://huggingface.co/gpt2/blob/main/config.json",
        "configs",
    ));
    let vocab_resource = Resource::Remote(RemoteResource::new( 
                //"https://huggingface.co/jweb/japanese-soseki-gpt2-1b/resolve/main/spiece.model".into(),
        //"https://huggingface.co/Narsil/gpt2/blob/main/vocab.json",
        //"https://huggingface.co/gpt2/blob/main/vocab.json",
        "https://huggingface.co/gpt2/blob/main/merges.txt",
        "vocab",
    ));
    let merges_resource = vocab_resource.clone();    
    let generate_config = GenerateConfig {        
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource, // not used        
        device: Device::Cpu,
        repetition_penalty: 1.6,
        min_length: 40,
        max_length: 128,
        do_sample: true,
        early_stopping: true,
        num_beams: 5,
        temperature: 1.0,
        top_k: 500,
        top_p: 0.95,
        ..Default::default()
    };
    /*
    let tokenizer = TokenizerOption::from_file(
        ModelType::T5,
        vocab_resource_token.get_local_path().unwrap().to_str().unwrap(),
        None,
        true,
        None,
        None,
    )?;
    */
    let mut gpt2_model = GPT2Generator::new(generate_config)?; //tokenizer.into())?;
    gpt2_model.set_device(Device::cuda_if_available());
    let input_text = "夏目漱石は、";
    let t1 = std::time::Instant::now();
    let output = gpt2_model.generate(Some(&[input_text]), None);
    println!("{}", output[0].text);
    println!("Elapsed Time(ms):{}",t1.elapsed().as_millis()); 
    Ok(())
}
