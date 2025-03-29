use std::fs::File;
use std::io;
use std::path::Path;
use std::str::FromStr;
use std::{env, error::Error, fs};

mod analysis;
mod search;
mod input;
mod lilypond;

use input::process_audio_segment;
//use search::genetic_search;


use crate::analysis::*;
use crate::search::*;
use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, formats::FormatOptions, io::MediaSourceStream,
    meta::MetadataOptions, probe::Hint,
};
//use std::path::Path;
//use std::fs::File;
use hound::{SampleFormat, WavSpec, WavWriter};

use bincode;
//use serde::{Deserialize, Serialize};

//use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

/// Mix and save the genetic search solution as a WAV file

// pub fn save_solution(
//     solution: &[CorpusVoices],
//     corpuses: &[Corpus],
//     output_path: &str,
// ) -> Result<(), Box<dyn Error>> {
//     // First, read all input files using symphonia
//     let mut all_samples: Vec<Vec<f32>> = Vec::new();

//     for voice in solution {
//         for &idx in &voice.entry_indices {
//             let path = &corpuses[voice.corpus_idx].entries[idx].path_name;

//             // Open the media source
//             let file = File::open(path)?;
//             let mss = MediaSourceStream::new(Box::new(file), Default::default());

//             // Create a hint to help the format registry guess what format reader is appropriate
//             let mut hint = Hint::new();
//             if let Some(extension) = Path::new(path).extension() {
//                 if let Some(ext_str) = extension.to_str() {
//                     hint.with_extension(ext_str);
//                 }
//             }

//             // Use the default options for metadata and format reading
//             let format_opts: FormatOptions = Default::default();
//             let metadata_opts: MetadataOptions = Default::default();

//             // Probe the media source
//             let probed =
//                 symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;

//             // Get the format reader
//             let mut format = probed.format;

//             // Get the default track
//             let track = format.default_track().ok_or("No default track found")?;

//             // Create a decoder for the track
//             let mut decoder = symphonia::default::get_codecs()
//                 .make(&track.codec_params, &DecoderOptions::default())?;

//             // Store decoded samples
//             let mut track_samples = Vec::new();

//             // Decode loop
//             while let Ok(packet) = format.next_packet() {
//                 // Decode the packet
//                 let decoded = decoder.decode(&packet)?;

//                 // Get the decoded audio samples
//                 let spec = *decoded.spec();
//                 let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
//                 sample_buf.copy_interleaved_ref(decoded);

//                 // If stereo, average the channels
//                 let samples = sample_buf.samples();
//                 if spec.channels.count() == 2 {
//                     for chunk in samples.chunks(2) {
//                         track_samples.push((chunk[0] + chunk[1]) * 0.5);
//                     }
//                 } else {
//                     track_samples.extend_from_slice(samples);
//                 }
//             }

//             all_samples.push(track_samples);
//         }
//     }

//     if all_samples.is_empty() {
//         return Err("No samples to mix".into());
//     }

//     // Convert to slice references for our mix function
//     let sample_slices: Vec<&[f32]> = all_samples.iter().map(|v| v.as_slice()).collect();

//     // Mix using our custom mix function
//     let mixed_samples = mix(&sample_slices);

//     // Create output WAV file
//     let spec = WavSpec {
//         channels: 1,
//         sample_rate: 44100, // You might want to get this from input files
//         bits_per_sample: 32,
//         sample_format: SampleFormat::Float,
//     };

//     let mut writer = WavWriter::create(output_path, spec)?;

//     // Write mixed samples
//     for &sample in &mixed_samples {
//         writer.write_sample(sample)?;
//     }

//     writer.finalize()?;
//     println!("Successfully saved mixed solution to {}", output_path);
//     Ok(())
// }

pub fn save_solution(
    solution: &[CorpusVoices],
    corpuses: &[Corpus],
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    // First, read all input files using symphonia
    let mut all_samples: Vec<Vec<f32>> = Vec::new();

    // Calculate number of voices for panning
    let num_voices = solution
        .iter()
        .map(|voice| voice.entry_indices.len())
        .sum::<usize>();

    // Create pan positions spread evenly from -1 (left) to 1 (right)
    let mut voice_count = 0;
    let mut pan_positions: Vec<f32> = Vec::new();

    for voice in solution {
        for _ in &voice.entry_indices {
            // Calculate pan position for this voice
            let pan = if num_voices > 1 {
                -1.0 + (2.0 * voice_count as f32 / (num_voices - 1) as f32)
            } else {
                0.0 // Center if only one voice
            };
            pan_positions.push(pan);
            voice_count += 1;
        }
    }

    voice_count = 0;
    for voice in solution {
        for &idx in &voice.entry_indices {
            let path = &corpuses[voice.corpus_idx].entries[idx].path_name;

            // Open the media source
            let file = File::open(path)?;
            let mss = MediaSourceStream::new(Box::new(file), Default::default());

            let mut hint = Hint::new();
            if let Some(extension) = Path::new(path).extension() {
                if let Some(ext_str) = extension.to_str() {
                    hint.with_extension(ext_str);
                }
            }

            let format_opts: FormatOptions = Default::default();
            let metadata_opts: MetadataOptions = Default::default();
            let probed =
                symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;
            let mut format = probed.format;
            let track = format.default_track().ok_or("No default track found")?;
            let mut decoder = symphonia::default::get_codecs()
                .make(&track.codec_params, &DecoderOptions::default())?;

            let mut track_samples = Vec::new();

            while let Ok(packet) = format.next_packet() {
                let decoded = decoder.decode(&packet)?;
                let spec = *decoded.spec();
                let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                sample_buf.copy_interleaved_ref(decoded);

                let samples = sample_buf.samples();
                if spec.channels.count() == 2 {
                    for chunk in samples.chunks(2) {
                        track_samples.push((chunk[0] + chunk[1]) * 0.5);
                    }
                } else {
                    track_samples.extend_from_slice(samples);
                }
            }

            all_samples.push(track_samples);
            voice_count += 1;
        }
    }

    if all_samples.is_empty() {
        return Err("No samples to mix".into());
    }

    // Mix samples with panning
    let mut mixed_left = vec![0.0f32; all_samples[0].len()];
    let mut mixed_right = vec![0.0f32; all_samples[0].len()];

    for (i, samples) in all_samples.iter().enumerate() {
        let pan = pan_positions[i];
        // Calculate left and right gains using equal power panning
        let left_gain = (std::f32::consts::PI * (pan + 1.0) / 4.0).cos();
        let right_gain = (std::f32::consts::PI * (pan + 1.0) / 4.0).sin();

        for (j, &sample) in samples.iter().enumerate() {
            if j < mixed_left.len() {
                mixed_left[j] += sample * left_gain;
                mixed_right[j] += sample * right_gain;
            }
        }
    }

    // Normalize if needed
    let max_amplitude = mixed_left
        .iter()
        .chain(mixed_right.iter())
        .map(|&x| x.abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(1.0);

    if max_amplitude > 1.0 {
        for sample in mixed_left.iter_mut() {
            *sample /= max_amplitude;
        }
        for sample in mixed_right.iter_mut() {
            *sample /= max_amplitude;
        }
    }

    // Create output WAV file (stereo)
    let spec = WavSpec {
        channels: 2,
        sample_rate: 44100,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(output_path, spec)?;

    // Write interleaved stereo samples
    for i in 0..mixed_left.len() {
        writer.write_sample(mixed_left[i])?;
        writer.write_sample(mixed_right[i])?;
    }

    writer.finalize()?;
    println!(
        "Successfully saved stereo mixed solution to {}",
        output_path
    );
    Ok(())
}

// pub fn save_solution(
//     solution: &[CorpusVoices],
//     corpuses: &[Corpus],
//     output_path: &str,
// ) -> Result<(), Box<dyn std::error::Error>> {
//     // First, read all input files to get their full samples and validate specs
//     let mut all_samples: Vec<Vec<f32>> = Vec::new();
//     let mut wav_spec: Option<WavSpec> = None;

//     for voice in solution {
//         for &idx in &voice.entry_indices {
//             let path = &corpuses[voice.corpus_idx].entries[idx].path_name;
//             let mut reader = WavReader::open(path)?;

//             // Get or validate WAV spec
//             let spec = reader.spec();
//             if let Some(existing_spec) = wav_spec {
//                 if spec != existing_spec {
//                     return Err("Inconsistent WAV specifications between input files".into());
//                 }
//             } else {
//                 wav_spec = Some(spec);
//             }

//             // Read all samples into memory
//             let samples: Vec<f32> = if spec.sample_format == SampleFormat::Float {
//                 reader.samples::<f32>().map(|s| s.unwrap()).collect()
//             } else {
//                 // Convert integer samples to float
//                 reader.samples::<i32>()
//                     .map(|s| s.unwrap() as f32 / i32::MAX as f32)
//                     .collect()
//             };

//             all_samples.push(samples);
//         }
//     }

//     if all_samples.is_empty() {
//         return Err("No samples to mix".into());
//     }

//     // Find the longest sample length
//     let max_length = all_samples.iter()
//         .map(|samples| samples.len())
//         .max()
//         .unwrap();

//     // Create output WAV spec
//     let spec = wav_spec.unwrap_or(WavSpec {
//         channels: 1,
//         sample_rate: 44100,
//         bits_per_sample: 32,
//         sample_format: SampleFormat::Float,
//     });

//     // Create output file
//     let mut writer = WavWriter::create(output_path, spec)?;

//     // Mix samples
//     for i in 0..max_length {
//         let mut mixed_sample = 0.0;
//         let mut active_sources = 0;

//         for samples in &all_samples {
//             if i < samples.len() {
//                 mixed_sample += samples[i];
//                 active_sources += 1;
//             }
//         }

//         // Normalize by number of active sources
//         if active_sources > 0 {
//             mixed_sample /= (active_sources as f32).sqrt();
//         }

//         // Ensure we don't exceed [-1.0, 1.0]
//         mixed_sample = mixed_sample.clamp(-1.0, 1.0);

//         writer.write_sample(mixed_sample)?;
//     }

//     writer.finalize()?;
//     println!("Successfully saved mixed solution to {}", output_path);
//     Ok(())
// }


/* 
fn print_solution_details(solution: &[CorpusVoices], corpuses: &[Corpus]) {
    println!("\nSolution Details:");
    println!("================");

    for voice in solution {
        let corpus = &corpuses[voice.corpus_idx];
        println!("\n{:?}:", corpus.instrument);
        println!("Number of voices: {}", voice.n_voices);

        // Convert MIDI notes to note names for display
        for (i, &idx) in voice.entry_indices.iter().enumerate() {
            let entry = &corpus.entries[idx];
            let note_name = midi_to_note_name(entry.info.midi_note);

            println!("Voice {}:", i + 1);
            println!("  Technique: {}", entry.info.technique);
            println!("  Pitch: {} (MIDI: {})", note_name, entry.info.midi_note);
            println!("  Dynamics: {}", entry.info.dynamics);
            if let Some(string_num) = entry.info.string {
                println!("  String: {}", string_num);
            }
            println!("  File: {}", entry.path_name);
        }

        // For string instruments with multiple voices, check if it's a double stop
        if (corpus.instrument == Instrument::Violin || corpus.instrument == Instrument::Cello)
            && voice.entry_indices.len() == 2
        {
            let entry1 = &corpus.entries[voice.entry_indices[0]];
            let entry2 = &corpus.entries[voice.entry_indices[1]];
            if can_play_double_stop(entry1, entry2, corpus.instrument) {
                println!("  (Valid double stop)");
            }
        }
    }
}
*/

// Helper function to convert MIDI note numbers to note names
fn midi_to_note_name(midi_note: u8) -> String {
    const NOTES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = (midi_note / 12) as i32 - 1;
    let note_index = (midi_note % 12) as usize;
    format!("{}{}", NOTES[note_index], octave)
}

fn main() -> io::Result<()> {

    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        eprintln!(
            "Usage: {} <target_file> <start_seconds> <end_seconds> <frame_duration> <output_file>",
            args[0]
        );
        eprintln!("The program expects:");
        eprintln!("  - A file 'corpus.txt' in the current directory");
        eprintln!("  - A target audio file to match");
        eprintln!("  - Start time in seconds");
        eprintln!("  - End time in seconds");
        eprintln!("  - Frame duration in seconds");
        eprintln!("  - An output path for the mixed solution");
        std::process::exit(1);
    }

    let target_file = &args[1];
    let start_seconds = match args[2].parse::<f32>() {
        Ok(seconds) => seconds,
        Err(_) => {
            eprintln!("Error: start_seconds must be a valid number");
            std::process::exit(1);
        }
    };
    let end_seconds = match args[3].parse::<f32>() {
        Ok(seconds) => seconds,
        Err(_) => {
            eprintln!("Error: end_seconds must be a valid number");
            std::process::exit(1);
        }
    };
    let frame_duration = match args[4].parse::<f32>() {
        Ok(seconds) => seconds,
        Err(_) => {
            eprintln!("Error: frame_duration must be a valid number");
            std::process::exit(1);
        }
    };

    let output_file = &args[5];

    // 1) Check if corpus.bincode exists
    let bincode_path = "corpus.bincode";
    let all_corpuses: Vec<Corpus> = if Path::new(bincode_path).exists() {
        // 2) If corpus.bincode is found, load from it
        println!("Found existing {bincode_path}, loading...");
        let file = File::open(bincode_path)?;
        match bincode::deserialize_from(file) {
            Ok(corp) => {
                println!("Successfully loaded corpus from bincode!");
                corp
            }
            Err(e) => {
                eprintln!("Failed to deserialize from {bincode_path}: {e}");
                std::process::exit(1);
            }
        }
    } else {
        // 3) Otherwise, read corpus.txt and parse it
        println!("No {bincode_path} found, reading corpus.txt...");

        let corpus_content = match std::fs::read_to_string("corpus.txt") {
            Ok(content) => content,
            Err(e) => {
                eprintln!("Failed to read corpus.txt: {}", e);
                std::process::exit(1);
            }
        };

        // Parse corpus sections
        let mut parsed_corpuses = Vec::new();
        let mut current_paths = Vec::new();
        let mut current_instrument: Option<Instrument> = None;

        for line in corpus_content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Check if this line is an instrument identifier
            if let Ok(instrument) = Instrument::from_str(line) {
                // If there's an existing corpus section pending, process it
                if !current_paths.is_empty() && current_instrument.is_some() {
                    process_corpus_section(
                        &mut parsed_corpuses,
                        &current_paths,
                        current_instrument.unwrap(),
                    )?;
                    current_paths.clear();
                }

                current_instrument = Some(instrument);
            } else {
                // Otherwise it's a path
                current_paths.push(line.to_string());
            }
        }

        // Process any leftover paths
        if !current_paths.is_empty() && current_instrument.is_some() {
            process_corpus_section(
                &mut parsed_corpuses,
                &current_paths,
                current_instrument.unwrap(),
            )?;
        }

        // Now that we've built parsed_corpuses, let's write it out to bincode
        println!("Writing corpus to {bincode_path} for future runs...");
        let file = File::create(bincode_path)?;
        if let Err(e) = bincode::serialize_into(file, &parsed_corpuses) {
            eprintln!("Failed to serialize corpuses to {bincode_path}: {e}");
            std::process::exit(1);
        }

        parsed_corpuses
    };

    
    /* 
    // Continue with the rest of your logic:
    println!("Loading target file: {target_file} at {start_seconds} seconds");
    let target = match load_samples_at_time(target_file, start_seconds) {
        Some(t) => {
            println!("Successfully loaded target file");
            t
        }
        None => {
            eprintln!("Failed to load target file: {}", target_file);
            std::process::exit(1);
        }
    };
*/

    println!("Loaded {} total corpuses", all_corpuses.len());
    for corpus in &all_corpuses {
        println!(
            "Corpus for {:?} contains {} entries",
            corpus.instrument,
            corpus.entries.len()
        );
    }

    // Process the audio segment
    if let Err(e) = process_audio_segment(target_file, start_seconds, end_seconds, &all_corpuses, frame_duration, output_file) {
        eprintln!("Error processing audio segment: {}", e);
        std::process::exit(1);
    }


    // let solution = genetic_search(&target, &all_corpuses, 3000, 1500, 50);

    // println!("\nGenetic search complete!");
    // print_solution_details(&solution, &all_corpuses);

    // println!("\nMixing solution...");
    // if let Err(e) = save_solution(&solution, &all_corpuses, output_file) {
    //     eprintln!("Failed to save mixed solution: {}", e);
    //     std::process::exit(1);
    // }

    Ok(())
}

// fn main() -> io::Result<()> {
//     let args: Vec<String> = env::args().collect();
//     if args.len() != 4 {
//         eprintln!(
//             "Usage: {} <target_file> <start_seconds> <output_file>",
//             args[0]
//         );
//         eprintln!("The program expects:");
//         eprintln!("  - A file 'corpus.txt' in the current directory");
//         eprintln!("  - A target audio file to match");
//         eprintln!("  - A start point in seconds within the target file");
//         eprintln!("  - An output path for the mixed solution");
//         std::process::exit(1);
//     }

//     let target_file = &args[1];
//     let start_seconds = match args[2].parse::<f32>() {
//         Ok(seconds) => seconds,
//         Err(_) => {
//             eprintln!("Error: start_seconds must be a valid number");
//             std::process::exit(1);
//         }
//     };
//     let output_file = &args[3];

//     // Read corpus file
//     let corpus_content = match std::fs::read_to_string("corpus.txt") {
//         Ok(content) => content,
//         Err(e) => {
//             eprintln!("Failed to read corpus.txt: {}", e);
//             std::process::exit(1);
//         }
//     };

//     // Parse corpus sections
//     let mut all_corpuses = Vec::new();
//     let mut current_paths = Vec::new();
//     let mut current_instrument: Option<Instrument> = None;

//     for line in corpus_content.lines() {
//         let line = line.trim();
//         if line.is_empty() {
//             continue;
//         }

//         // Check if this line is an instrument identifier
//         if let Ok(instrument) = Instrument::from_str(line) {
//             // Process previous section if it exists
//             if !current_paths.is_empty() && current_instrument.is_some() {
//                 process_corpus_section(
//                     &mut all_corpuses,
//                     &current_paths,
//                     current_instrument.unwrap(),
//                 )?;
//                 current_paths.clear();
//             }

//             // Start new section
//             current_instrument = Some(instrument);
//             continue;
//         }

//         // If we get here, this line is a path
//         current_paths.push(line.to_string());
//     }

//     // Process the last section
//     if !current_paths.is_empty() && current_instrument.is_some() {
//         process_corpus_section(
//             &mut all_corpuses,
//             &current_paths,
//             current_instrument.unwrap(),
//         )?;
//     }

//     println!("Loaded {} total corpuses", all_corpuses.len());
//     for corpus in &all_corpuses {
//         println!(
//             "Corpus for {:?} contains {} entries",
//             corpus.instrument,
//             corpus.entries.len()
//         );
//     }

//     // Load target file with specified start time
//     println!(
//         "Loading target file: {} at {} seconds",
//         target_file, start_seconds
//     );
//     let target = match load_samples_at_time(target_file, start_seconds) {
//         Some(target_input) => {
//             println!("Successfully loaded target file");
//             target_input
//         }
//         None => {
//             eprintln!("Failed to load target file: {}", target_file);
//             std::process::exit(1);
//         }
//     };

//     let solution = genetic_search(&target, &all_corpuses, 100, 50, 10);

//     println!("\nGenetic search complete!");

//     print_solution_details(&solution, &all_corpuses);

//     println!("\nMixing solution...");

//     if let Err(e) = save_solution(&solution, &all_corpuses, output_file) {
//         eprintln!("Failed to save mixed solution: {}", e);
//         std::process::exit(1);
//     }

//     Ok(())
// }

fn get_audio_files_from_dir(dir_path: &str) -> io::Result<Vec<String>> {
    let mut audio_files = Vec::new();

    for entry in fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();

        // Check if it's a file and has an audio extension
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if let Some(ext_str) = ext.to_str() {
                    // Add more audio extensions as needed
                    if ["wav", "mp3", "aiff", "flac"].contains(&ext_str.to_lowercase().as_str()) {
                        if let Some(path_str) = path.to_str() {
                            audio_files.push(path_str.to_string());
                        }
                    }
                }
            }
        }
    }

    Ok(audio_files)
}

fn process_corpus_section(
    all_corpuses: &mut Vec<Corpus>,
    paths: &[String],
    instrument: Instrument,
) -> io::Result<()> {
    println!("Loading corpus for {:?}...", instrument);

    // Collect all files from directories
    let mut all_files = Vec::new();
    for path in paths {
        match get_audio_files_from_dir(path) {
            Ok(mut files) => all_files.append(&mut files),
            Err(e) => {
                eprintln!("Warning: Failed to read directory {}: {}", path, e);
                continue;
            }
        }
    }

    if all_files.is_empty() {
        eprintln!("Warning: No audio files found in paths: {:?}", paths);
        return Ok(());
    }

    let entries = load_audio_corpus(&all_files, instrument);
    println!(
        "Successfully loaded {} files for {:?}",
        entries.len(),
        instrument
    );

    let corpus = Corpus {
        entries,
        name: format!("{:?}", instrument),
        instrument,
    };

    all_corpuses.push(corpus);

    Ok(())
}
// fn main() -> io::Result<()> {
//     // Get command line arguments
//     let args: Vec<String> = env::args().collect();
//     if args.len() != 2 {
//         eprintln!("Usage: {} <target_file>", args[0]);
//         eprintln!("The program expects a file 'corpus.txt' in the current directory");
//         std::process::exit(1);
//     }

//     let target_file = &args[1];

//     // Read corpus file list
//     let corpus_paths = match read_lines("corpus.txt") {
//         Ok(paths) => paths,
//         Err(e) => {
//             eprintln!("Failed to read corpus.txt: {}", e);
//             std::process::exit(1);
//         }
//     };

//     // Filter out empty lines and whitespace
//     let corpus_paths: Vec<String> = corpus_paths
//         .into_iter()
//         .map(|s| s.trim().to_string())
//         .filter(|s| !s.is_empty())
//         .collect();

//     println!("Loading corpus of {} files...", corpus_paths.len());
//     let corpus = load_audio_corpus(&corpus_paths);
//     println!("Successfully loaded {} files from corpus", corpus.len());

//     // Load target file
//     println!("Loading target file: {}", target_file);
//     let target;
//     match load_middle_n_samples(target_file) {
//         Some(target_input) => {
//             target = target_input;
//             println!("Successfully loaded target file");
//             //println!("Target samples: {}", target.len());
//             // Here you could do something with the target and corpus...
//         }
//         None => {
//             eprintln!("Failed to load target file: {}", target_file);
//             std::process::exit(1);
//         }
//     }

//     let result = genetic_search(&target, &corpus, 3, 1000, 6, 100);
//     println!("Result: {:?}", result);
//     /*
//     for entry in &corpus {
//         let distance = compare_audio_segments(&target, &entry.samples);
//         println!(
//             "Distance: {}, entry pitch: {}",
//             distance, entry.info.midi_note
//         );
//     }
//     */
//     /*
//        for (i, f) in feature_vector(&target).iter().enumerate() {
//     println!("Feature {}: {}", i, f);
//      }
//      */
//     Ok(())
// }
