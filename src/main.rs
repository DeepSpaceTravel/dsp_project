#![allow(warnings)]
use cpal::{Data, traits::{DeviceTrait, HostTrait}};
const SAMPLE_RATE: u32 = 48000;

fn init() {
    // Device
    let host = cpal::default_host();
    let devices = host.devices().unwrap();
    for device in devices{
        // println!("{:?}", device.description().unwrap().name());
        // println!();
    }

    let input_device = host.default_input_device().expect("no input device available");
    let output_device = host.default_output_device().expect("no output device available");

    // Device config
    let mut supported_input_configs_range = input_device.supported_input_configs().expect("error while querying configs");
    // Query max sample rate
    println!("{:?}", supported_input_configs_range.next().expect("no supported config!?"));
    
    let supported_input_config = supported_input_configs_range.next().expect("no supported config!?").with_max_sample_rate();
    // let supported_input_config = supported_input_configs_range.next().expect("no supported config!?").with_sample_rate(SAMPLE_RATE);
    println!("{:?}", supported_input_config);
    let input_config = supported_input_config.config();
    println!("{:?}", input_config);

    // Data
    // let input_data = ;

    // Stream
    // let input_stream = input_config;
    // let input_stream = input_device.build_input_stream(
    //     &input_config,
    //     move |,
    //     eprintln!("`data_callback` Error occurred"),
    //     None
    // );
}

fn main() {
    // println!("Hello, world!");
    init();
}
