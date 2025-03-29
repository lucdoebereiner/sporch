//use realfft::num_traits::Float;
use realfft::{num_complex, RealFftPlanner};
//use num_complex::Complex32;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs::File;
use std::path::Path;
use std::str::FromStr;
use symphonia::core::audio::AudioBufferRef;
use symphonia::core::audio::Signal;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;



use serde::{Deserialize, Serialize};
//use bincode;

pub const TARGET_SAMPLE_RATE: u32 = 48000;
pub const TARGET_SAMPLES: usize = 16384;

/// Compute a Hanning window of a given size.
fn hanning_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| 0.5 * (1.0 - ((2.0 * PI * n as f32) / (size as f32 - 1.0)).cos()))
        .collect()
}

/// Mix multiple audio slices
pub fn mix(slices: &[&[f32]]) -> Vec<f32> {
    if slices.is_empty() {
        panic!("At least one slice is required.");
    }

    // Determine the minimum length of the slices to ensure proper mixing
    let min_length = slices.iter().map(|slice| slice.len()).min().unwrap_or(0);
    if min_length == 0 {
        panic!("Slices must not be empty.");
    }

    // Mix slices by averaging them with power-based scaling
    let mut mixed_slice = vec![0.0; min_length];
    for slice in slices {
        for i in 0..min_length {
            mixed_slice[i] += slice[i];
        }
    }

    // Normalize by the square root of the number of layers
    let scaling_factor = (slices.len() as f32).sqrt();
    for sample in &mut mixed_slice {
        *sample /= scaling_factor;
    }

    // Process the mixed slice to compute the feature vector
    // process_audio_from_slice(&mixed_slice, weight)
    mixed_slice
}

/// Process a given audio slice and return the normalized magnitude spectrum and descriptors.
pub fn feature_vector(samples: &[f32]) -> Vec<f32> {
    let window_size = TARGET_SAMPLES;

    // Select the middle  samples (or as much as available if the slice is smaller)
    let slice = if samples.len() >= window_size {
        let middle = samples.len() / 2;
        let start_index = middle.saturating_sub(window_size / 2);
        &samples[start_index..(start_index + window_size)]
    } else {
        panic!(
            "The input slice must be at least {} samples long",
            window_size
        );
    };

    // Apply Hanning window
    let hanning = hanning_window(window_size);
    let windowed_samples: Vec<f32> = slice
        .iter()
        .zip(hanning.iter())
        .map(|(s, w)| s * w)
        .collect();

    // Perform FFT
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(window_size);
    let mut spectrum = vec![num_complex::Complex32::new(0.0, 0.0); window_size / 2 + 1];
    let mut input = windowed_samples.clone();
    r2c.process(&mut input, &mut spectrum).unwrap();

    // Compute magnitude spectrum and normalize it
    let magnitude_spectrum: Vec<f32> = spectrum
        .iter()
        .map(|c| c.norm() / window_size as f32) // Normalize with respect to frame size
        .collect();

    /*
        // Log-scaling of magnitude values
        const EPSILON: f32 = 1e-10; // Small constant to avoid log(0)
        const REFERENCE_VALUE: f32 = 1e-5; // A fixed value to shift everything uniformly

        for mag in &mut magnitude_spectrum {
            *mag = (*mag + EPSILON).ln() - REFERENCE_VALUE.ln(); // Fixed shift based on a constant reference
        }
    */
  /* 
    magnitude_spectrum[0] = 0.0;
    const EPSILON: f32 = 1e-10; // to avoid log(0)
    for mag in &mut magnitude_spectrum {
        *mag = 20.0 * (*mag + EPSILON).log10();
    }
    */
    /*
    // Normalize magnitudes to [0, 1]
    let mag_max = magnitude_spectrum.iter().cloned().fold(f32::MIN, f32::max);
    if mag_max > 0.0 {
        for mag in &mut magnitude_spectrum {
            *mag /= mag_max;
        }
    }
    */

    // Compute spectral descriptors (from unnormalized spectrum)
    //   let spectral_centroid = compute_spectral_centroid(&spectrum);
    // let spectral_spread = compute_spectral_spread(&spectrum, spectral_centroid);
    //let spectral_flatness = compute_spectral_flatness(&spectrum);

    // Normalize descriptors to [0, 1]
    //let normalized_centroid = spectral_centroid / window_size as f32;
    //let normalized_spread = spectral_spread / window_size as f32;
    //let normalized_flatness = spectral_flatness;

    // Combine features into one vector with weight scaling
  //  let mut feature_vector = vec![];

    // Add weighted magnitude spectrum
    //feature_vector.extend(magnitude_spectrum);

    // Add weighted and normalized descriptors
    //   feature_vector.push(normalized_centroid);
    // feature_vector.push(normalized_spread);
    //feature_vector.push(normalized_flatness);

    //feature_vector
    magnitude_spectrum
}

/* 
/// Compute the spectral centroid
fn compute_spectral_centroid(spectrum: &[num_complex::Complex32]) -> f32 {
    let numerator: f32 = spectrum
        .iter()
        .enumerate()
        .map(|(i, c)| i as f32 * c.norm())
        .sum();
    let denominator: f32 = spectrum.iter().map(|c| c.norm()).sum();
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Compute the spectral spread
fn compute_spectral_spread(spectrum: &[num_complex::Complex32], centroid: f32) -> f32 {
    let numerator: f32 = spectrum
        .iter()
        .enumerate()
        .map(|(i, c)| ((i as f32 - centroid).powi(2)) * c.norm())
        .sum();
    let denominator: f32 = spectrum.iter().map(|c| c.norm()).sum();
    if denominator > 0.0 {
        (numerator / denominator).sqrt()
    } else {
        0.0
    }
}

/// Compute the spectral flatness
fn compute_spectral_flatness(spectrum: &[num_complex::Complex32]) -> f32 {
    let n = spectrum.len() as f32;
    // Sum the natural logs of the norms (avoiding log(0) with max)
    let sum_logs: f32 = spectrum.iter().map(|c| (c.norm().max(1e-10)).ln()).sum();
    let geometric_mean = (sum_logs / n).exp();

    let arithmetic_mean = spectrum.iter().map(|c| c.norm()).sum::<f32>() / n;
    if arithmetic_mean > 0.0 {
        geometric_mean / arithmetic_mean
    } else {
        0.0
    }
}
*/

/* 
fn normalize_db_spectrum(spectrum: &[f32]) -> Vec<f32> {
    // Assume spectrum values are in the range [-200, 0] dB.
    spectrum
        .iter()
        .map(|&db| (db.clamp(-200.0, 0.0) + 200.0) / 200.0)
        .collect()
}
        */

/*         
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}
    */

/// Normalize by RMS to remove overall loudness differences.
fn normalize_by_rms(magnitudes: &mut [f32]) {
    let rms = magnitudes.iter().map(|&x| x * x).sum::<f32>().sqrt();
    // Avoid divide-by-zero
    if rms > 1e-12 {
        for x in magnitudes.iter_mut() {
            *x /= rms;
        }
    }
}

fn compare_spectra_with_normalization(
    target_features: &[f32],
    candidate_features: &[f32],
    num_magnitude: usize,
) -> f32 {
    // Only take the first num_magnitude values from each feature vector.
    let mut target_mags = &mut target_features[..num_magnitude].to_vec();
    let mut candidate_mags = &mut candidate_features[..num_magnitude].to_vec();
    normalize_by_rms(&mut target_mags);
    normalize_by_rms(&mut candidate_mags);
    //let norm_target = 
    //let norm_candidate = 
    /*
        for (i, e) in norm_candidate.iter().enumerate() {
            println!("i : {}, e: {}", i, e);
        }
    */

    efficient_asymmetric_similarity(&target_mags, &candidate_mags)
   // cosine_similarity(&target_mags, &candidate_mags)
}


fn efficient_asymmetric_similarity(target: &[f32], candidate: &[f32]) -> f32 {
    // Standard dot product calculation
    let mut dot_product = 0.0;
    let mut target_squared_sum = 0.0;
    let mut candidate_squared_sum = 0.0;
    let mut excess_energy_sum = 0.0;
    
    // Pre-calculate the maximum target value for normalization
    let target_max = target.iter().cloned().fold(0.0, f32::max);
    let threshold = target_max * 0.05;
    
    for i in 0..target.len() {
        // Standard cosine similarity components
        dot_product += target[i] * candidate[i];
        target_squared_sum += target[i] * target[i];
        candidate_squared_sum += candidate[i] * candidate[i];
        
        // Calculate excess energy (asymmetric penalty)
        if candidate[i] > target[i] && target[i] < threshold {
            excess_energy_sum += (candidate[i] - target[i]) / target_max;
        }
    }
    
    // Calculate cosine similarity
    let cosine = if target_squared_sum > 0.0 && candidate_squared_sum > 0.0 {
        dot_product / (target_squared_sum.sqrt() * candidate_squared_sum.sqrt())
    } else {
        0.0
    };
    
    // Apply penalty (normalized by spectrum size)
    let penalty = (excess_energy_sum / target.len() as f32).min(0.5);
    
    // Final score with asymmetric penalty
    cosine * (1.0 - penalty)
}

/* *
fn hybrid_spectral_similarity(target: &[f32], candidate: &[f32]) -> f32 {
    // Standard cosine similarity
    let cosine_sim = cosine_similarity(target, candidate);
    
    // Calculate asymmetric penalty for extra content
    let mut extra_content_penalty = 0.0;
    let mut significant_bins = 0;
    
    for i in 0..target.len() {
        // Find bins where candidate has more energy than target
        if candidate[i] > target[i] {
            // Calculate the excess energy (normalized by max target magnitude)
            let target_max = target.iter().cloned().fold(0.0, f32::max);
            let excess = (candidate[i] - target[i]) / target_max;
            extra_content_penalty += excess;
        }
        
        // Count significant target bins for normalization
        if target[i] > 0.05 * target.iter().cloned().fold(0.0, f32::max) {
            significant_bins += 1;
        }
    }
    
    // Normalize penalty
    if significant_bins > 0 {
        extra_content_penalty /= significant_bins as f32;
    }
    
    // Calculate peak matching component
    let target_peaks = find_significant_peaks(target, 8);
    let candidate_peaks = find_significant_peaks(candidate, 12); // Get more candidate peaks
    
    let peak_match_score = calculate_peak_match(
        &target_peaks, 
        &candidate_peaks, 
        target.len()
    );
    
    // Combine scores (adjust weights to taste)
    // Penalize extra content by reducing the cosine similarity
    let adjusted_cosine = cosine_sim * (1.0 - extra_content_penalty.min(0.7));
    
    0.6 * adjusted_cosine + 0.4 * peak_match_score
}

fn find_significant_peaks(spectrum: &[f32], max_peaks: usize) -> Vec<(usize, f32)> {
    let threshold = 0.1 * spectrum.iter().cloned().fold(0.0, f32::max);
    let mut peaks = Vec::new();
    
    // First, find all peaks above threshold
    for i in 2..spectrum.len()-2 {
        if spectrum[i] > threshold &&
           spectrum[i] > spectrum[i-1] &&
           spectrum[i] > spectrum[i-2] &&
           spectrum[i] > spectrum[i+1] &&
           spectrum[i] > spectrum[i+2] {
            peaks.push((i, spectrum[i]));
        }
    }
    
    // Sort by magnitude (highest first) and take top N
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    peaks.truncate(max_peaks);
    
    peaks
}

fn calculate_peak_match(
    target_peaks: &[(usize, f32)],
    candidate_peaks: &[(usize, f32)],
    spectrum_length: usize
) -> f32 {
    if target_peaks.is_empty() {
        return 0.0;
    }
    
    let mut total_match = 0.0;
    
    for &(target_idx, target_mag) in target_peaks {
        let mut best_match = 0.0;
        
        for &(cand_idx, cand_mag) in candidate_peaks {
            // Calculate frequency distance (normalized)
            let freq_distance = (target_idx as f32 - cand_idx as f32).abs() / 
                                (spectrum_length as f32 * 0.1); // 10% of spectrum is "close"
            
            // Calculate magnitude similarity
            let mag_ratio = if target_mag > cand_mag {
                cand_mag / target_mag  // Penalize if candidate peak is weaker
            } else {
                1.0 - 0.3 * ((cand_mag / target_mag) - 1.0).min(1.0)  // Slight penalty if too strong
            };
            
            // Combine to get match quality for this peak
            let match_quality = (1.0 - freq_distance.min(1.0)) * mag_ratio;
            best_match = best_match.max(match_quality);
        }
        
        total_match += best_match;
    }
    
    // Calculate mean match quality across all target peaks
    total_match / target_peaks.len() as f32
}
    */

/* 
fn adaptive_spectrum_comparison(target: &[f32], candidate: &[f32]) -> f32 {
    // Calculate where energy is concentrated in target
    let total_energy: f32 = target.iter().map(|&x| x*x).sum();
    let low_end = target.len() / 10;
    let mid_end = target.len() * 4 / 10;
    
    let low_energy = target[0..low_end].iter().map(|&x| x*x).sum::<f32>() / total_energy;
    let mid_energy = target[low_end..mid_end].iter().map(|&x| x*x).sum::<f32>() / total_energy;
    let high_energy = target[mid_end..].iter().map(|&x| x*x).sum::<f32>() / total_energy;
    
    // Weight similarity based on energy distribution
    let low_sim = cosine_similarity(&target[0..low_end], &candidate[0..low_end]) * low_energy;
    let mid_sim = cosine_similarity(&target[low_end..mid_end], &candidate[low_end..mid_end]) * mid_energy;
    let high_sim = cosine_similarity(&target[mid_end..], &candidate[mid_end..]) * high_energy;
    
    low_sim + mid_sim + high_sim
}
*/
// fn weighted_spectrum_comparison(target: &[f32], candidate: &[f32]) -> f32 {
//     // Balance low/mid/high frequency contributions
//     let low_freq_weight = 0.3;
//     let mid_freq_weight = 0.4;
//     let high_freq_weight = 0.3;
    
//     let low_end = target.len() / 10;
//     let mid_end = target.len() * 4 / 10;
    
//     let low_sim = cosine_similarity(&target[0..low_end], &candidate[0..low_end]) * low_freq_weight;
//     let mid_sim = cosine_similarity(&target[low_end..mid_end], &candidate[low_end..mid_end]) * mid_freq_weight;
//     let high_sim = cosine_similarity(&target[mid_end..], &candidate[mid_end..]) * high_freq_weight;
    
//     low_sim + mid_sim + high_sim
// }

/* 
pub fn compare_audio_segments(target: &[f32], candidate: &[f32]) -> f32 {
    // Create feature vectors from the audio slices.
    let target_features = feature_vector(target);
    let candidate_features = feature_vector(candidate);

    //  println!("target_features: {}", target_features.len());

    // For a window size of 16384, the FFT yields (16384 / 2 + 1) magnitude features.
    let num_magnitude = (TARGET_SAMPLES / 2) + 1; // equals 8193

    // Ensure that the feature vectors are the same length.
    let total_features = target_features.len();
    if candidate_features.len() != total_features {
        panic!("Feature vector lengths do not match!");
    }

  
    compare_spectra_with_normalization(&target_features, &candidate_features, num_magnitude)
  
}
*/


pub fn compare_audio_segments_with_precomputed_target(
    target_features: &[f32],
    candidate: &[f32]
) -> f32 {
    // Create feature vector from the candidate audio slice
    let candidate_features = feature_vector(candidate);

    // For a window size of 16384, the FFT yields (16384 / 2 + 1) magnitude features.
    let num_magnitude = (TARGET_SAMPLES / 2) + 1; // equals 8193

    // Ensure that the feature vectors are the same length
    let total_features = target_features.len();
    if candidate_features.len() != total_features {
        panic!("Feature vector lengths do not match!");
    }

    compare_spectra_with_normalization(target_features, &candidate_features, num_magnitude)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Eq, Hash)]
pub enum Instrument {
    Violin,
    Cello,
    Accordion,
    Synth,
}

impl FromStr for Instrument {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Vn" => Ok(Instrument::Violin),
            "Vc" => Ok(Instrument::Cello),
            "Acc" => Ok(Instrument::Accordion),
            "Synth" => Ok(Instrument::Accordion),
            _ => Err(()),
        }
    }
}

fn has_strings(instrument: Instrument) -> bool {
    match instrument {
        Instrument::Violin | Instrument::Cello => true,
        _ => false,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PlayingTechnique {
    Ordinario,
    ArtificialHarmonic,
    SulPonticello,
}


impl std::fmt::Display for PlayingTechnique {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PlayingTechnique::Ordinario => write!(f, "ord"),
            PlayingTechnique::ArtificialHarmonic => write!(f, "art_harm"),
            PlayingTechnique::SulPonticello => write!(f, "pont"),
        }
    }
}

impl FromStr for PlayingTechnique {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ord" => Ok(PlayingTechnique::Ordinario),
            "art_harm" => Ok(PlayingTechnique::ArtificialHarmonic),
            "pont" => Ok(PlayingTechnique::SulPonticello),
            _ => Err(()),
        }
    }
}



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusEntryInfo {
    //pub instrument: Instrument,
    pub technique: PlayingTechnique,
    pub midi_note: u8,
    pub dynamics: String,
    pub string: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusEntry {
    pub samples: Vec<f32>,
    pub path_name: String,
    pub info: CorpusEntryInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Corpus {
    pub entries: Vec<CorpusEntry>,
    pub name: String,
    pub instrument: Instrument,
}




/// Returns true if two bowed-string CorpusEntries (violin or cello)
/// can plausibly be played as a double stop on adjacent strings.
///
/// Assumptions / Simplifications:
/// - String numbering: 1 = highest pitch, 4 = lowest pitch
/// - Each note's MIDI pitch must be >= that string's open pitch
///   and <= some arbitrary max fingerboard pitch.
/// - We only allow adjacent-string double stops (|string1 - string2| == 1).
/// - We limit the interval size to a "max_interval" (hand stretch), ignoring
///   that higher positions allow bigger intervals.
///
/// Adjust numeric limits as needed for your corpus or performance practice.
pub fn can_play_double_stop(
    e1: &CorpusEntry,
    e2: &CorpusEntry,
    instrument: Instrument,
) -> bool {
    // Make sure we have string info
    let s1 = match e1.info.string {
        Some(s) => s,
        None => return false,
    };
    let s2 = match e2.info.string {
        Some(s) => s,
        None => return false,
    };

    // Must be on adjacent strings:
    if (s1 as i8 - s2 as i8).abs() != 1 {
        return false;
    }

    if e1.info.technique != PlayingTechnique::Ordinario || e2.info.technique != PlayingTechnique::Ordinario {
        return false;
    }

    if e1.info.dynamics != e2.info.dynamics {
        return false;
    }   
    // Retrieve the MIDI notes
    let note1 = e1.info.midi_note;
    let note2 = e2.info.midi_note;

    // Define open-string MIDI pitches (string1 through string4),
    // a rough maximum pitch, and a rough max interval in semitones.
    // (You can tweak these numbers!)
    let (open_pitches, max_pitch, min_interval, max_interval) = match instrument {
        Instrument::Violin => {
            // Violin standard tuning: E5 (76), A4 (69), D4 (62), G3 (55)
            // For simplicity, let's set a max around C7 (96).
            // We'll allow up to a 17- or 18-semitone stretch as a rough upper bound.
            ([76_u8, 69, 62, 55], 96_u8, 1, 14)
        }
        Instrument::Cello => {
            // Cello standard tuning (highest=1 => A3=57, D3=50, G2=43, C2=36)
            // We'll cap at ~C6 (84) for demonstration.
            // We'll allow up to ~15 semitones as a typical "comfort" limit in lower positions.
            ([57_u8, 50, 43, 36], 84_u8, 3, 11)
        }
        // If it's not violin or cello, return false, or handle other instruments...
        _ => return false,
    };

    // Helper closure to check if a single note is playable on the given string:
    let is_note_playable_on_string = |midi_note: u8, string_number: u8| {
        // Convert string_number (1..=4) to array index (0..=3).
        let idx = (string_number - 1) as usize;
        let open_pitch = open_pitches[idx];
        // Must be >= open pitch and <= max_pitch
        (midi_note >= open_pitch) && (midi_note <= max_pitch)
    };

    // Check that each note is in range for its string
    if !is_note_playable_on_string(note1, s1) {
        return false;
    }
    if !is_note_playable_on_string(note2, s2) {
        return false;
    }

    // Finally, check the interval: if it's too large for a typical hand stretch in any position,
    // we return false.  This is a big simplification—real players can manage bigger intervals
    // in higher positions, or might find even smaller intervals difficult in low positions, etc.
    let interval = note2 as i16 - note1 as i16;
    if interval > max_interval as i16 || interval < min_interval as i16 {
        return false;
    }

    // If all checks pass, assume it's playable as a double stop
    true
}


fn parse_corpus_filename(filename: &str, instrument: Instrument) -> Option<CorpusEntryInfo> {
    // Create note to MIDI number mapping
    let mut note_to_midi: HashMap<&str, u8> = HashMap::new();
    let notes = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    for (i, note) in notes.iter().enumerate() {
        note_to_midi.insert(note, i as u8);
    }

    // Extract the filename from the path
    let basename = filename.split('/').last()?;

    // Split the filename by hyphens
    let parts: Vec<&str> = basename.split('-').collect();
    if parts.len() < 4 {
        return None;
    }

    // Extract instrument (first part)
    let _instrument = parts[0].to_string();

    // Extract technique (second part)
    let technique : PlayingTechnique = PlayingTechnique::from_str(parts[1]).unwrap();

    // Parse the note (third part)
    let note_str = parts[2];
    // Split note into note name and octave
    let note_name: String = note_str.chars().take_while(|c| !c.is_numeric()).collect();
    let octave: i32 = note_str
        .chars()
        .skip_while(|c| !c.is_numeric())
        .collect::<String>()
        .parse()
        .ok()?;

    // Calculate MIDI note number
    // MIDI note 60 is middle C (C4)
    let base_note = *note_to_midi.get(note_name.as_str())?;
    let midi_note = base_note as i32 + (octave + 1) * 12;
    if !(0..=127).contains(&midi_note) {
        return None;
    }

    // Extract dynamics (fourth part)
    let dynamics = parts[3].to_string();
    // Extract the string number if the instrument type is BowedString
    let string = if has_strings(instrument) {
        parts.get(4).and_then(|s| s.strip_suffix('c')).and_then(|s| s.parse().ok())
    } else {
        None
    };

    Some(CorpusEntryInfo {
        //instrument,
        technique,
        midi_note: midi_note as u8,
        dynamics,
        string,
    })
}

/// Load the middle N samples from a list of .mp3 files (left channel only).
pub fn load_audio_corpus(paths: &[String], instrument: Instrument) -> Vec<CorpusEntry> {
    // Create three iterators:
    // 1. The original paths
    // 2. Sample data from paths
    // 3. Corpus info from paths
    paths
        .iter()
        .zip(paths.iter().map(load_middle_n_samples))
        .zip(paths.iter().map(|s| parse_corpus_filename(s.as_str(), instrument)))
        // Flatten the nested zip and filter out Nones
        .filter_map(
            |((path, maybe_samples), maybe_info)| match (maybe_samples, maybe_info) {
                (Some(samples), Some(info)) => Some(CorpusEntry {
                    samples,
                    path_name: path.clone(),
                    info,
                }),
                _ => None,
            },
        )
        .collect()
}

// Helper: Convert a 24-bit sample (stored as i32) to an f32 value in the range [-1.0, 1.0].
pub fn convert_s24_to_f32(sample: i32) -> f32 {
    // 2^(24-1) = 2^23 = 8388608.0
    sample as f32 / 8388608.0
}

pub fn load_middle_n_samples(path: &String) -> Option<Vec<f32>> {
    let file = File::open(Path::new(path)).ok()?;

    // Create appropriate hint based on file extension
    let mut hint = Hint::new();
    if path.to_lowercase().ends_with(".mp3") {
        hint.with_extension("mp3");
    } else if path.to_lowercase().ends_with(".wav") {
        hint.with_extension("wav");
    } else {
        eprintln!("Unsupported file format: {}", path);
        return None;
    }

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .ok()?;

    let mut format = probed.format;
    let track = format.default_track()?;
    let codec_params = &track.codec_params;

    let source_sample_rate = codec_params.sample_rate?;
    let num_channels = codec_params.channels?.count();

    if num_channels == 0 || num_channels > 2 {
        eprintln!("Unsupported number of channels: {}", num_channels);
        return None;
    }

    // For MP3s, we might not have n_frames, so calculate based on duration if available
    let total_frames = codec_params.n_frames.or_else(|| {
        codec_params
            .time_base
            .map(|tb| (source_sample_rate as f64 / tb.denom as f64) as u64)
    })?;

    //  println!("total_frames: {}", total_frames);

    let middle_frame = total_frames / 2;

    // Adjust sample count based on source sample rate to get correct duration
    let source_num_samples = if source_sample_rate != TARGET_SAMPLE_RATE {
        ((TARGET_SAMPLES as f64) * (source_sample_rate as f64) / (TARGET_SAMPLE_RATE as f64))
            as usize
    } else {
        TARGET_SAMPLES
    };

    let frames_needed = source_num_samples as u64;
    let start_frame = middle_frame.saturating_sub(frames_needed / 2);

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .ok()?;

    let mut samples: Vec<f32> = Vec::with_capacity(source_num_samples);
    let mut current_frame = 0;

    // Skip packets until we reach our target frame
    while current_frame < start_frame {
        match format.next_packet() {
            Ok(packet) => {
                if let Ok(decoded) = decoder.decode(&packet) {
                    current_frame += decoded.capacity() as u64;
                }
            }
            Err(_) => return None,
        }
    }

    // Read only the frames we need
    while samples.len() < source_num_samples {
        match format.next_packet() {
            Ok(packet) => {
                if let Ok(decoded) = decoder.decode(&packet) {
                    match decoded {
                        AudioBufferRef::F32(buf) => {
                            if num_channels == 2 {
                                for frame in 0..buf.capacity() {
                                    let mono = (buf.chan(0)[frame] + buf.chan(1)[frame]) * 0.5;
                                    samples.push(mono);
                                }
                            } else {
                                samples.extend_from_slice(buf.chan(0));
                            }
                        }
                        AudioBufferRef::U8(buf) => {
                            if num_channels == 2 {
                                for frame in 0..buf.capacity() {
                                    let mono = ((buf.chan(0)[frame] as f32 / 255.0 - 0.5)
                                        + (buf.chan(1)[frame] as f32 / 255.0 - 0.5))
                                        * 0.5;
                                    samples.push(mono);
                                }
                            } else {
                                samples
                                    .extend(buf.chan(0).iter().map(|&s| (s as f32 / 255.0) - 0.5));
                            }
                        }
                        AudioBufferRef::S24(buf) => {
                            if num_channels == 2 {
                                for frame in 0..buf.capacity() {
                                    let sample0 = convert_s24_to_f32(buf.chan(0)[frame].inner());
                                    let sample1 = convert_s24_to_f32(buf.chan(1)[frame].inner());
                                    let mono = (sample0 + sample1) * 0.5;
                                    samples.push(mono);
                                }
                            } else {
                                samples.extend(buf.chan(0).iter().map(|&s| convert_s24_to_f32(s.inner())));
                            }
                        }
                        AudioBufferRef::S16(buf) => {
                            if num_channels == 2 {
                                for frame in 0..buf.capacity() {
                                    let mono = ((buf.chan(0)[frame] as f32 / 32768.0)
                                        + (buf.chan(1)[frame] as f32 / 32768.0))
                                        * 0.5;
                                    samples.push(mono);
                                }
                            } else {
                                samples.extend(buf.chan(0).iter().map(|&s| s as f32 / 32768.0));
                            }
                        }
                        AudioBufferRef::S32(buf) => {
                            if num_channels == 2 {
                                for frame in 0..buf.capacity() {
                                    let mono = ((buf.chan(0)[frame] as f32 / 2147483648.0)
                                        + (buf.chan(1)[frame] as f32 / 2147483648.0))
                                        * 0.5;
                                    samples.push(mono);
                                }
                            } else {
                                samples
                                    .extend(buf.chan(0).iter().map(|&s| s as f32 / 2147483648.0));
                            }
                        }
                        _ => return None,
                    }
                }
            }
            Err(_) => break,
        }
    }

    // Ensure we have enough samples
    if samples.len() < source_num_samples {
        eprintln!(
            "Warning: File {} has fewer than {} samples.",
            path, source_num_samples
        );
        return None;
    }

    //samples.truncate(source_num_samples);

    //   println!("source sample rate: {}", source_sample_rate);
    //  println!("target sample rate: {}", TARGET_SAMPLE_RATE);

    // println!("samples: {}, source num samples: {}", samples.len(), source_num_samples);

    // Resample if necessary
    let final_samples = if source_sample_rate != TARGET_SAMPLE_RATE {
        let mut params = SincFixedIn::<f32>::new(
            TARGET_SAMPLE_RATE as f64 / source_sample_rate as f64,
            2.0,
            SincInterpolationParameters {
                sinc_len: 1024,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Cubic,
                oversampling_factor: 128,
                window: WindowFunction::BlackmanHarris2,
            },
            samples.len(),
            1,
        )
        .ok()?;

        let waves_in = vec![samples];
        params.process(&waves_in, None).ok()?[0].to_vec()
    } else {
        samples
    };

    // println!("final_samples: {}", final_samples.len());

    // Ensure we have exactly TARGET_SAMPLES after resampling
    let mut final_samples = final_samples;
    final_samples.truncate(TARGET_SAMPLES);

    if final_samples.len() < TARGET_SAMPLES {
        eprintln!(
            "Warning: After resampling, file {} has fewer than {} samples.",
            path, TARGET_SAMPLES
        );
        return None;
    }

    Some(final_samples)
}

/* 
// In analysis.rs, modify load_middle_n_samples to load_samples_at_time:
pub fn load_samples_at_time(path: &String, start_seconds: f32) -> Option<Vec<f32>> {
    let file = File::open(Path::new(path)).ok()?;

    let mut hint = Hint::new();
    if path.to_lowercase().ends_with(".mp3") {
        hint.with_extension("mp3");
    } else if path.to_lowercase().ends_with(".wav") {
        hint.with_extension("wav");
    } else {
        eprintln!("Unsupported file format: {}", path);
        return None;
    }

    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .ok()?;

    let mut format = probed.format;
    let track = format.default_track()?;
    let codec_params = &track.codec_params;
    let source_sample_rate = codec_params.sample_rate?;

    let num_channels = codec_params.channels?.count();

    if num_channels == 0 || num_channels > 2 {
        eprintln!("Unsupported number of channels: {}", num_channels);
        return None;
    }

    
    // Calculate the start frame based on time
    let start_frame = (start_seconds * source_sample_rate as f32) as u64;
    
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .ok()?;

    // Adjust sample count based on source sample rate
    let source_num_samples = if source_sample_rate != TARGET_SAMPLE_RATE {
        ((TARGET_SAMPLES as f64) * (source_sample_rate as f64) / (TARGET_SAMPLE_RATE as f64))
            as usize
    } else {
        TARGET_SAMPLES
    };

    let mut samples: Vec<f32> = Vec::with_capacity(source_num_samples);
    let mut current_frame = 0;

    // Skip packets until we reach our target start time
    while current_frame < start_frame {
        match format.next_packet() {
            Ok(packet) => {
                if let Ok(decoded) = decoder.decode(&packet) {
                    current_frame += decoded.capacity() as u64;
                }
            }
            Err(_) => return None,
        }
    }

     // Read only the frames we need
     while samples.len() < source_num_samples {
        match format.next_packet() {
            Ok(packet) => {
                if let Ok(decoded) = decoder.decode(&packet) {
                    match decoded {
                        AudioBufferRef::F32(buf) => {
                            if num_channels == 2 {
                                for frame in 0..buf.capacity() {
                                    let mono = (buf.chan(0)[frame] + buf.chan(1)[frame]) * 0.5;
                                    samples.push(mono);
                                }
                            } else {
                                samples.extend_from_slice(buf.chan(0));
                            }
                        }
                        AudioBufferRef::U8(buf) => {
                            if num_channels == 2 {
                                for frame in 0..buf.capacity() {
                                    let mono = ((buf.chan(0)[frame] as f32 / 255.0 - 0.5)
                                        + (buf.chan(1)[frame] as f32 / 255.0 - 0.5))
                                        * 0.5;
                                    samples.push(mono);
                                }
                            } else {
                                samples
                                    .extend(buf.chan(0).iter().map(|&s| (s as f32 / 255.0) - 0.5));
                            }
                        }
                        AudioBufferRef::S24(buf) => {
                            if num_channels == 2 {
                                for frame in 0..buf.capacity() {
                                    let sample0 = convert_s24_to_f32(buf.chan(0)[frame].inner());
                                    let sample1 = convert_s24_to_f32(buf.chan(1)[frame].inner());
                                    let mono = (sample0 + sample1) * 0.5;
                                    samples.push(mono);
                                }
                            } else {
                                samples.extend(buf.chan(0).iter().map(|&s| convert_s24_to_f32(s.inner())));
                            }
                        }
                        AudioBufferRef::S16(buf) => {
                            if num_channels == 2 {
                                for frame in 0..buf.capacity() {
                                    let mono = ((buf.chan(0)[frame] as f32 / 32768.0)
                                        + (buf.chan(1)[frame] as f32 / 32768.0))
                                        * 0.5;
                                    samples.push(mono);
                                }
                            } else {
                                samples.extend(buf.chan(0).iter().map(|&s| s as f32 / 32768.0));
                            }
                        }
                        AudioBufferRef::S32(buf) => {
                            if num_channels == 2 {
                                for frame in 0..buf.capacity() {
                                    let mono = ((buf.chan(0)[frame] as f32 / 2147483648.0)
                                        + (buf.chan(1)[frame] as f32 / 2147483648.0))
                                        * 0.5;
                                    samples.push(mono);
                                }
                            } else {
                                samples
                                    .extend(buf.chan(0).iter().map(|&s| s as f32 / 2147483648.0));
                            }
                        }
                        _ => return None,
                    }
                }
            }
            Err(_) => break,
        }
    }

    // Ensure we have enough samples
    if samples.len() < source_num_samples {
        eprintln!(
            "Warning: File {} has fewer than {} samples at the specified time.",
            path, source_num_samples
        );
        return None;
    }

    // Resample if necessary
    let final_samples = if source_sample_rate != TARGET_SAMPLE_RATE {
        let mut params = SincFixedIn::<f32>::new(
            TARGET_SAMPLE_RATE as f64 / source_sample_rate as f64,
            2.0,
            SincInterpolationParameters {
                sinc_len: 1024,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Cubic,
                oversampling_factor: 128,
                window: WindowFunction::BlackmanHarris2,
            },
            samples.len(),
            1,
        )
        .ok()?;

        let waves_in = vec![samples];
        params.process(&waves_in, None).ok()?[0].to_vec()
    } else {
        samples
    };

    let mut final_samples = final_samples;
    final_samples.truncate(TARGET_SAMPLES);

    if final_samples.len() < TARGET_SAMPLES {
        eprintln!(
            "Warning: After resampling, file {} has fewer than {} samples.",
            path, TARGET_SAMPLES
        );
        return None;
    }

    Some(final_samples)
}
    */