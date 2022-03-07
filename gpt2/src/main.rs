// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

extern crate anyhow;

use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};

// コマンドラインから
fn input() -> String {
    println!("Input: ");
    let mut text = String::new();
    std::io::stdin().read_line(&mut text).unwrap();
    return text.trim().to_string();
}

fn main() -> anyhow::Result<()> {
    //    Set-up model
    println!("Loading model...");
    let generate_config = TextGenerationConfig {
        model_type: ModelType::GPT2,
        max_length: 50,
        do_sample: false,
        num_beams: 1,
        temperature: 1.0,
        num_return_sequences: 1,
        repetition_penalty: 1.6,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config)?;
    
    println!("Ready to run. Type QUIT to escape.\n\n");
    loop {
        let input_context = input();
        // QUITで終了できるように
        if input_context == "QUIT" { break; }
        // 時間測定スタート
        let start = std::time::Instant::now();
        println!("Generating...");
        let output = model.generate(&[input_context], None);

        for sentence in output {
            println!("「{:?}」", sentence);
        }
        // 時間測定。差分を取る
        let stop = std::time::Instant::now();
        println!("<Time: {:.3}s>", (stop.duration_since(start).as_millis() as f64) / 1000.0);
        println!("\n")
    };
    return Ok(());
}
