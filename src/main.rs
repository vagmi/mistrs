use anyhow::Result;
use indexmap::IndexMap;
use either::Either;
use std::sync::Arc;
use mistralrs::{NormalLoaderBuilder, NormalSpecificConfig, NormalLoaderType, MistralRs,
                GGUFLoaderBuilder, GGUFSpecificConfig,
                MistralRsBuilder, Device, SchedulerMethod, Request, Response, NormalRequest, 
                RequestMessage, SamplingParams, Constraint, TokenSource, ModelDType, DeviceMapMetadata, GgmlDType};
use tokio::sync::mpsc::channel;

fn setup_isq() -> Result<Arc<MistralRs>> {
    let device = Device::new_metal(0)?;
    let loader = NormalLoaderBuilder::new(
        NormalSpecificConfig {
            use_flash_attn: false,
            repeat_last_n: 64
        },
        None,
        None,
        Some("mistralai/Mistral-7B-Instruct-v0.1".to_string())

    ).build(NormalLoaderType::Mistral);
    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        &ModelDType::Auto,
        &device,
        false,
        DeviceMapMetadata::dummy(),
        Some(GgmlDType::Q4K)
    )?;
    let mrs = MistralRsBuilder::new(pipeline, SchedulerMethod::Fixed(5.try_into().unwrap())).build();

    Ok(mrs)
}

fn setup_quant() -> Result<Arc<MistralRs>> {
    let device = Device::new_metal(0)?;
    let loader = GGUFLoaderBuilder::new(
        GGUFSpecificConfig {
            repeat_last_n: 64
        },
        None,
        Some("argilla/CapybaraHermes-2.5-Mistral-7B".to_string()),
        "TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF".to_string(),
        "capybarahermes-2.5-mistral-7b.Q4_K_M.gguf".to_string()
    ).build();
    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        &ModelDType::Auto,
        &device,
        false,
        DeviceMapMetadata::dummy(),
        None,
    )?;
    let mrs = MistralRsBuilder::new(pipeline, SchedulerMethod::Fixed(5.try_into().unwrap())).build();
    Ok(mrs)
}

fn main() -> Result<()> {
    let mrs = setup_quant()?;
    let (tx, mut rx) = channel(10_000);
    let request = Request::Normal(NormalRequest {
        messages: RequestMessage::Chat(vec![
                      IndexMap::from([
                        ("role".to_string(), Either::Left("user".to_string())),
                        ("content".to_string(), Either::Left("Can you write a poem about the nine planets solar system? Include a hat tip to pluto even though it is not a planet.".to_string()))
                      ])
        ]),
        sampling_params: SamplingParams::default(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        id: 0,
        constraint: Constraint::None,
        suffix: None,
        adapters: None
    });
    mrs.get_sender()?.blocking_send(request)?;
    let rsp = rx.blocking_recv().unwrap();
    match rsp {
        Response::Done(c) => println!("Text: {}\nT/s: {},\nC/s: {}", c.choices[0].message.content, c.usage.avg_prompt_tok_per_sec, c.usage.avg_compl_tok_per_sec ),

        Response::InternalError(e) => panic!("Internal Error: {}", e),
        Response::ValidationError(e) => panic!("Internal Error: {}", e),
        Response::ModelError(e, c) => panic!(
            "Model error: {e}. Response: Text: {}, Prompt T/s: {}, Completion T/s: {}",
            c.choices[0].message.content,
            c.usage.avg_prompt_tok_per_sec,
            c.usage.avg_compl_tok_per_sec
        ),
        _ => unreachable!(),
    }

    Ok(())
}
