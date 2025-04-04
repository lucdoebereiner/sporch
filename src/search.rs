use rand::prelude::*;
use std::cmp::Ordering;

use crate::analysis::*;

#[derive(Clone, Debug)]
struct Individual {
    entries: Vec<CorpusVoices>, // Indices into the corpus
    fitness: f32,
}

impl PartialOrd for Individual {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.fitness.partial_cmp(&other.fitness)
    }
}

impl PartialEq for Individual {
    fn eq(&self, other: &Self) -> bool {
        self.fitness == other.fitness
    }
}

/// Get the allowed number of simultaneous notes for each instrument
fn get_allowed_notes(instrument: Instrument) -> Vec<usize> {
    match instrument {
        Instrument::Violin | Instrument::Cello => vec![0, 1, 1, 1, 2], // Can play either 1 or 2 notes
        Instrument::Accordion | Instrument::Synth => (0..=6).collect(),                 // Can play 1 to 6 notes
    }
}

/// Represents how many notes each corpus contributes to the solution
#[derive(Clone, Debug)]
pub struct CorpusVoices {
    pub corpus_idx: usize,         // Which corpus this refers to
    pub n_voices: usize,           // How many notes this corpus contributes
    pub entry_indices: Vec<usize>, // Indices into the corpus entries
}

/// Generates a random selection of unique indices
fn get_unique_indices(count: usize, max_index: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut indices = Vec::with_capacity(count);
    let mut available: Vec<usize> = (0..max_index).collect();

    while indices.len() < count && !available.is_empty() {
        let idx = rng.random_range(0..available.len());
        indices.push(available.remove(idx));
    }

    indices
}

/// Try to find a valid double stop for string instruments
fn find_valid_double_stop(
    corpus: &Corpus,
    rng: &mut impl Rng,
    max_attempts: usize,
) -> Option<Vec<usize>> {
    for _ in 0..max_attempts {
        let idx1 = rng.random_range(0..corpus.entries.len());
        let idx2 = rng.random_range(0..corpus.entries.len());

        if can_play_double_stop(
            &corpus.entries[idx1],
            &corpus.entries[idx2],
            corpus.instrument,
        ) {
            return Some(vec![idx1, idx2]);
        }
    }
    None
}

/*
fn get_unique_indices_with_matching_dynamics(
    count: usize,
    corpus: &Corpus,
    rng: &mut impl Rng,
) -> Vec<usize> {
    // First, group entries by dynamics
    let mut dynamics_groups: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();

    for (idx, entry) in corpus.entries.iter().enumerate() {
        dynamics_groups
            .entry(entry.info.dynamics.clone())
            .or_insert_with(Vec::new)
            .push(idx);
    }

    // Select a random dynamics group that has enough entries
    let valid_groups: Vec<&String> = dynamics_groups
        .iter()
        .filter(|(_, indices)| indices.len() >= count)
        .map(|(dynamics, _)| dynamics)
        .collect();

    if valid_groups.is_empty() {
        // Fallback: if no dynamics group has enough entries, just return what we can
        return get_unique_indices(count.min(corpus.entries.len()), corpus.entries.len(), rng);
    }

    // Pick a random valid dynamics group
    let chosen_dynamics = valid_groups.choose(rng).unwrap();
    let available_indices = &dynamics_groups[*chosen_dynamics];

    // Get random unique indices from this group
    let mut indices = Vec::with_capacity(count);
    let mut remaining: Vec<usize> = available_indices.clone();

    while indices.len() < count && !remaining.is_empty() {
        let idx = rng.random_range(0..remaining.len());
        indices.push(remaining.remove(idx));
    }

    indices
}
*/

fn get_unique_indices_with_unique_pitches_and_matching_dynamics(
    count: usize,
    corpus: &Corpus,
    rng: &mut impl Rng,
) -> Vec<usize> {
    // First, group entries by dynamics
    let mut dynamics_groups: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();

    for (idx, entry) in corpus.entries.iter().enumerate() {
        dynamics_groups
            .entry(entry.info.dynamics.clone())
            .or_insert_with(Vec::new)
            .push(idx);
    }

    // Select a random dynamics group that has enough entries
    let valid_groups: Vec<&String> = dynamics_groups
        .iter()
        .filter(|(_, indices)| indices.len() >= count)
        .map(|(dynamics, _)| dynamics)
        .collect();

    if valid_groups.is_empty() {
        // Fallback: if no dynamics group has enough entries, just return what we can
        return get_unique_indices(count.min(corpus.entries.len()), corpus.entries.len(), rng);
    }

    // Pick a random valid dynamics group
    let chosen_dynamics = valid_groups.choose(rng).unwrap();
    let available_indices = &dynamics_groups[*chosen_dynamics];

    // NEW: Ensure pitch uniqueness
    let mut used_pitches = std::collections::HashSet::new();
    let mut indices = Vec::with_capacity(count);

    // Create a copy of available indices to draw from
    let mut remaining = available_indices.clone();
    // Shuffle it to get random selection
    remaining.shuffle(rng);

    for &idx in &remaining {
        let pitch = corpus.entries[idx].info.midi_note;

        if !used_pitches.contains(&pitch) {
            used_pitches.insert(pitch);
            indices.push(idx);

            if indices.len() >= count {
                break;
            }
        }
    }

    indices
}

fn generate_corpus_voices(corpus: &Corpus, corpus_idx: usize, rng: &mut impl Rng) -> CorpusVoices {
    let allowed_notes = get_allowed_notes(corpus.instrument);
    let n_voices = *allowed_notes.choose(rng).unwrap();

    let entry_indices = match corpus.instrument {
        Instrument::Accordion => {
            // For accordion, ensure matching dynamics and unique pitches
            get_unique_indices_with_unique_pitches_and_matching_dynamics(n_voices, corpus, rng)
        }
        Instrument::Violin | Instrument::Cello if n_voices == 2 => {
            // Try to find a valid double stop, fall back to single note if can't find one
            match find_valid_double_stop(corpus, rng, 500) {
                Some(indices) => indices,
                None => get_unique_indices(1, corpus.entries.len(), rng),
            }
        }
        _ => {
            // For other cases, use original unique indices function
            get_unique_indices(n_voices, corpus.entries.len(), rng)
        }
    };

    CorpusVoices {
        corpus_idx,
        n_voices: entry_indices.len(),
        entry_indices,
    }
}

fn get_indices_with_unique_pitches_from_parents(
    parent_voice: &CorpusVoices,
    corpus: &Corpus,
    n_voices: usize,
    rng: &mut impl Rng,
) -> Vec<usize> {
    // Get the dynamics of the parent entries
    let parent_dynamics: Vec<&String> = parent_voice
        .entry_indices
        .iter()
        .map(|&idx| &corpus.entries[idx].info.dynamics)
        .collect();

    // If parent had multiple entries, they should have the same dynamics
    let target_dynamics = parent_dynamics[0];

    // Find all entries with matching dynamics
    let matching_indices: Vec<usize> = (0..corpus.entries.len())
        .filter(|&idx| &corpus.entries[idx].info.dynamics == target_dynamics)
        .collect();

    if matching_indices.len() >= n_voices {
        // NEW: Ensure pitch uniqueness
        let mut used_pitches = std::collections::HashSet::new();
        let mut indices = Vec::with_capacity(n_voices);

        // Create a shuffled copy of matching indices
        let mut available = matching_indices.clone();
        available.shuffle(rng);

        for &idx in &available {
            let pitch = corpus.entries[idx].info.midi_note;

            if !used_pitches.contains(&pitch) {
                used_pitches.insert(pitch);
                indices.push(idx);

                if indices.len() >= n_voices {
                    break;
                }
            }
        }

        if indices.len() == n_voices {
            return indices;
        }
    }

    // Fallback: use original parent indices if we can't find enough unique pitches
    parent_voice.entry_indices.clone()
}

// /// Generate a random valid voice configuration for a corpus
// fn generate_corpus_voices(
//     corpus: &Corpus,
//     corpus_idx: usize,
//     rng: &mut impl Rng
// ) -> CorpusVoices {
//     let allowed_notes = get_allowed_notes(corpus.instrument);
//     let n_voices = *allowed_notes.choose(rng).unwrap();

//     let entry_indices = if (corpus.instrument == Instrument::Violin
//                           || corpus.instrument == Instrument::Cello)
//                          && n_voices == 2 {
//         // Try to find a valid double stop, fall back to single note if can't find one
//         match find_valid_double_stop(corpus, rng, 500) {
//             Some(indices) => indices,
//             None => get_unique_indices(1, corpus.entries.len(), rng)
//         }
//     } else {
//         // need to have the same dynamics
//         get_unique_indices(n_voices, corpus.entries.len(), rng)
//     };

//     CorpusVoices {
//         corpus_idx,
//         n_voices: entry_indices.len(),
//         entry_indices,
//     }
// }

/// Calculate fitness for a given set of voices
// fn calculate_fitness(voices: &[CorpusVoices], corpuses: &[Corpus], target: &[f32]) -> f32 {
//     let mut all_samples = Vec::new();
//     for voice in voices {
//         for &idx in &voice.entry_indices {
//             all_samples.push(&corpuses[voice.corpus_idx].entries[idx].samples);
//         }
//     }
//     let mixed = mix(&all_samples.iter().map(|v| v.as_slice()).collect::<Vec<_>>());
//     compare_audio_segments_with_precomputed_target(target, &mixed)
// }

/// Calculate fitness for a given set of voices
fn calculate_fitness(voices: &[CorpusVoices], corpuses: &[Corpus], target: &[f32]) -> f32 {
    let mut all_samples = Vec::new();
    for voice in voices {
        // Skip voices with 0 voices (instruments not participating)
        if voice.n_voices == 0 || voice.entry_indices.is_empty() {
            continue;
        }
        
        for &idx in &voice.entry_indices {
            all_samples.push(&corpuses[voice.corpus_idx].entries[idx].samples);
        }
    }
    
    // If no instruments were selected, return a very low fitness
    // This should never happen due to checks in crossover and mutate,
    // but just in case it does, make sure these solutions are heavily penalized
    if all_samples.is_empty() {
        return -1.0;  // Using a negative fitness ensures these solutions are always worse than any real solution
    }
    
    let mixed = mix(&all_samples.iter().map(|v| v.as_slice()).collect::<Vec<_>>());
    compare_audio_segments_with_precomputed_target(target, &mixed)
}

/* 
fn get_indices_with_matching_dynamics_from_parents(
    parent_voice: &CorpusVoices,
    corpus: &Corpus,
    n_voices: usize,
    rng: &mut impl Rng,
) -> Vec<usize> {
    // Get the dynamics of the parent entries
    let parent_dynamics: Vec<&String> = parent_voice
        .entry_indices
        .iter()
        .map(|&idx| &corpus.entries[idx].info.dynamics)
        .collect();

    // If parent had multiple entries, they should have the same dynamics
    let target_dynamics = parent_dynamics[0];

    // Find all entries with matching dynamics
    let matching_indices: Vec<usize> = (0..corpus.entries.len())
        .filter(|&idx| &corpus.entries[idx].info.dynamics == target_dynamics)
        .collect();

    if matching_indices.len() >= n_voices {
        // Get random unique indices from matching ones
        let mut indices = Vec::with_capacity(n_voices);
        let mut available = matching_indices.clone();

        while indices.len() < n_voices && !available.is_empty() {
            let idx = rng.random_range(0..available.len());
            indices.push(available.remove(idx));
        }
        indices
    } else {
        // Fallback: use original parent indices if we can't find enough matching ones
        parent_voice.entry_indices.clone()
    }
}
*/

// fn crossover(
//     parent1: &Individual,
//     parent2: &Individual,
//     corpuses: &[Corpus],
//     target: &[f32],
//     rng: &mut impl Rng,
// ) -> Individual {
//     let mut child_voices = Vec::new();

//     for corpus_idx in 0..corpuses.len() {
//         let corpus = &corpuses[corpus_idx];
//         let p1_voice = &parent1.entries[corpus_idx];
//         let p2_voice = &parent2.entries[corpus_idx];

//         match corpus.instrument {
//             Instrument::Violin | Instrument::Cello => {
//                 let is_p1_double = p1_voice.n_voices == 2;
//                 let is_p2_double = p2_voice.n_voices == 2;

//                 // 25% chance to try creating a new double stop from both parents' entries
//                 if (is_p1_double || is_p2_double) && rng.random_bool(0.25) {
//                     let mut possible_pairs = Vec::new();

//                     // Collect all possible pairs from both parents' entries
//                     let p1_entries = &p1_voice.entry_indices;
//                     let p2_entries = &p2_voice.entry_indices;

//                     for &idx1 in p1_entries {
//                         for &idx2 in p2_entries {
//                             if idx1 != idx2
//                                 && can_play_double_stop(
//                                     &corpus.entries[idx1],
//                                     &corpus.entries[idx2],
//                                     corpus.instrument,
//                                 )
//                             {
//                                 possible_pairs.push(vec![idx1, idx2]);
//                             }
//                         }
//                     }

//                     if !possible_pairs.is_empty() {
//                         let chosen_pair = possible_pairs.choose(rng).unwrap();
//                         child_voices.push(CorpusVoices {
//                             corpus_idx,
//                             n_voices: 2,
//                             entry_indices: chosen_pair.clone(),
//                         });
//                         continue;
//                     }
//                 }

//                 // If we didn't create a new double stop, randomly choose between parents
//                 let chosen_parent = if rng.random_bool(0.5) {
//                     p1_voice
//                 } else {
//                     p2_voice
//                 };

//                 if chosen_parent.n_voices == 1 {
//                     let other_parent = if chosen_parent as *const _ == p1_voice as *const _ {
//                         p2_voice
//                     } else {
//                         p1_voice
//                     };

//                     let entry_pool: Vec<usize> = chosen_parent
//                         .entry_indices
//                         .iter()
//                         .chain(other_parent.entry_indices.iter())
//                         .copied()
//                         .collect();

//                     let selected_idx = *entry_pool.choose(rng).unwrap();
//                     child_voices.push(CorpusVoices {
//                         corpus_idx,
//                         n_voices: 1,
//                         entry_indices: vec![selected_idx],
//                     });
//                 } else {
//                     child_voices.push(chosen_parent.clone());
//                 }
//             }

//             Instrument::Accordion | Instrument::Synth => {
//                 // Choose parent to inherit from

//                 // Choose parent to inherit from
//                 let chosen_parent = if rng.random_bool(0.5) {
//                     p1_voice
//                 } else {
//                     p2_voice
//                 };

//                 // Get new indices with matching dynamics and unique pitches
//                 let n_voices = chosen_parent.n_voices;
//                 let entry_indices = get_indices_with_unique_pitches_from_parents(
//                     chosen_parent,
//                     corpus,
//                     n_voices,
//                     rng,
//                 );

//                 child_voices.push(CorpusVoices {
//                     corpus_idx,
//                     n_voices: entry_indices.len(),
//                     entry_indices,
//                 });
//             }
//         }
//     }

//     let fitness = calculate_fitness(&child_voices, corpuses, target);
//     Individual {
//         entries: child_voices,
//         fitness,
//     }
// }


fn crossover(
    parent1: &Individual,
    parent2: &Individual,
    corpuses: &[Corpus],
    target: &[f32],
    rng: &mut impl Rng,
) -> Individual {
    let mut child_voices = Vec::new();
    let mut any_instrument_included = false;

    // First, determine which instruments to include
    let mut include_instruments = Vec::with_capacity(corpuses.len());
    
    for corpus_idx in 0..corpuses.len() {
        let p1_voice = &parent1.entries[corpus_idx];
        let p2_voice = &parent2.entries[corpus_idx];
        
        // Determine if this instrument will be included in the child
        let include_instrument = match (p1_voice.n_voices > 0, p2_voice.n_voices > 0) {
            (true, true) => true,  // Both parents use this instrument
            (false, false) => rng.random_bool(0.1),  // Small chance to add if neither parent uses it
            (true, false) | (false, true) => rng.random_bool(0.7),  // High chance to inherit if one parent uses it
        };
        
        include_instruments.push(include_instrument);
        if include_instrument {
            any_instrument_included = true;
        }
    }
    
    // If no instruments would be included, force at least one random instrument
    if !any_instrument_included {
        let forced_idx = rng.random_range(0..corpuses.len());
        include_instruments[forced_idx] = true;
    }
    
    // Now process each instrument
    for corpus_idx in 0..corpuses.len() {
        let corpus = &corpuses[corpus_idx];
        let p1_voice = &parent1.entries[corpus_idx];
        let p2_voice = &parent2.entries[corpus_idx];
        
        // Use our pre-determined inclusion decision
        let include_instrument = include_instruments[corpus_idx];
        
        if !include_instrument {
            // If not including, add an empty voice
            child_voices.push(CorpusVoices {
                corpus_idx,
                n_voices: 0,
                entry_indices: Vec::new(),
            });
            continue;
        }
        
        // Now handle instruments that are included based on their type
        match corpus.instrument {
            Instrument::Violin | Instrument::Cello => {
                // Use the original code for string instruments, but only if they're participating
                let p1_participating = p1_voice.n_voices > 0;
                let p2_participating = p2_voice.n_voices > 0;
                
                if !p1_participating && !p2_participating {
                    // This shouldn't happen given our logic above, but just to be safe
                    child_voices.push(generate_corpus_voices(corpus, corpus_idx, rng));
                    continue;
                }
                
                // If only one parent is participating, use that one
                if !p1_participating {
                    child_voices.push(p2_voice.clone());
                    continue;
                }
                if !p2_participating {
                    child_voices.push(p1_voice.clone());
                    continue;
                }
                
                // Otherwise, both parents are participating
                let is_p1_double = p1_voice.n_voices == 2;
                let is_p2_double = p2_voice.n_voices == 2;

                // Rest of the original string instrument crossover logic
                // 25% chance to try creating a new double stop from both parents' entries
                if (is_p1_double || is_p2_double) && rng.random_bool(0.25) {
                    // Original double stop creation logic...
                    let mut possible_pairs = Vec::new();
                    let p1_entries = &p1_voice.entry_indices;
                    let p2_entries = &p2_voice.entry_indices;

                    for &idx1 in p1_entries {
                        for &idx2 in p2_entries {
                            if idx1 != idx2
                                && can_play_double_stop(
                                    &corpus.entries[idx1],
                                    &corpus.entries[idx2],
                                    corpus.instrument,
                                )
                            {
                                possible_pairs.push(vec![idx1, idx2]);
                            }
                        }
                    }

                    if !possible_pairs.is_empty() {
                        let chosen_pair = possible_pairs.choose(rng).unwrap();
                        child_voices.push(CorpusVoices {
                            corpus_idx,
                            n_voices: 2,
                            entry_indices: chosen_pair.clone(),
                        });
                        continue;
                    }
                }

                // If we didn't create a new double stop, randomly choose between parents
                let chosen_parent = if rng.random_bool(0.5) {
                    p1_voice
                } else {
                    p2_voice
                };
                child_voices.push(chosen_parent.clone());
            },

            Instrument::Accordion | Instrument::Synth => {
                // Handle keyboard instruments (accordion and synth)
                let p1_participating = p1_voice.n_voices > 0;
                let p2_participating = p2_voice.n_voices > 0;
                
                if !p1_participating && !p2_participating {
                    // Generate a new voice
                    child_voices.push(generate_corpus_voices(corpus, corpus_idx, rng));
                    continue;
                }
                
                // Choose parent to inherit from (or generate new if only one is participating)
                let chosen_parent = if !p1_participating {
                    p2_voice
                } else if !p2_participating {
                    p1_voice
                } else if rng.random_bool(0.5) {
                    p1_voice
                } else {
                    p2_voice
                };

                // Get new indices with matching dynamics and unique pitches
                let n_voices = chosen_parent.n_voices;
                let entry_indices = get_indices_with_unique_pitches_from_parents(
                    chosen_parent,
                    corpus,
                    n_voices,
                    rng,
                );

                child_voices.push(CorpusVoices {
                    corpus_idx,
                    n_voices: entry_indices.len(),
                    entry_indices,
                });
            }
        }
    }

    let fitness = calculate_fitness(&child_voices, corpuses, target);
    Individual {
        entries: child_voices,
        fitness,
    }
}

// fn mutate(
//     parent: &Individual,
//     corpuses: &[Corpus],
//     target: &[f32],
//     rng: &mut impl Rng,
// ) -> Individual {
//     let mut child_voices = parent.entries.clone();

//     // Pick a random corpus to mutate
//     let corpus_idx = rng.random_range(0..corpuses.len());

//     // Simply generate a new random voice configuration for this corpus
//     child_voices[corpus_idx] = generate_corpus_voices(&corpuses[corpus_idx], corpus_idx, rng);

//     let fitness = calculate_fitness(&child_voices, corpuses, target);
//     Individual {
//         entries: child_voices,
//         fitness,
//     }
// }

fn mutate(
    parent: &Individual,
    corpuses: &[Corpus],
    target: &[f32],
    rng: &mut impl Rng,
) -> Individual {
    let mut child_voices = parent.entries.clone();

    // 20% chance to add/remove an instrument entirely
    if rng.random_bool(0.2) {
        // Pick a random corpus to toggle participation
        let corpus_idx = rng.random_range(0..corpuses.len());
        
        // If this instrument is currently not participating (0 voices), add it
        if child_voices[corpus_idx].n_voices == 0 {
            child_voices[corpus_idx] = generate_corpus_voices(&corpuses[corpus_idx], corpus_idx, rng);
            // Ensure we generate non-zero voices
            while child_voices[corpus_idx].n_voices == 0 {
                child_voices[corpus_idx] = generate_corpus_voices(&corpuses[corpus_idx], corpus_idx, rng);
            }
        } 
        // Otherwise, remove it (set to 0 voices) - but only if it wouldn't leave us with no instruments
        else {
            // Check if this is the only instrument that's currently participating
            let participating_count = child_voices.iter().filter(|v| v.n_voices > 0).count();
            
            if participating_count > 1 {
                // Safe to remove this instrument
                child_voices[corpus_idx] = CorpusVoices {
                    corpus_idx,
                    n_voices: 0,
                    entry_indices: Vec::new(),
                };
            } else {
                // This is the only instrument - pick a different one to add instead
                let mut other_indices: Vec<usize> = (0..corpuses.len())
                    .filter(|&i| i != corpus_idx)
                    .collect();
                
                if !other_indices.is_empty() {
                    let new_idx = *other_indices.choose(rng).unwrap();
                    child_voices[new_idx] = generate_corpus_voices(&corpuses[new_idx], new_idx, rng);
                    // Ensure we generate non-zero voices
                    while child_voices[new_idx].n_voices == 0 {
                        child_voices[new_idx] = generate_corpus_voices(&corpuses[new_idx], new_idx, rng);
                    }
                }
            }
        }
    } 
    // 80% chance to do regular mutation
    else {
        // Pick a random corpus to mutate that is currently participating
        let participating_indices: Vec<usize> = child_voices.iter()
            .enumerate()
            .filter(|(_, voice)| voice.n_voices > 0)
            .map(|(idx, _)| idx)
            .collect();
        
        // If no instruments are participating, add one
        if participating_indices.is_empty() {
            let corpus_idx = rng.random_range(0..corpuses.len());
            child_voices[corpus_idx] = generate_corpus_voices(&corpuses[corpus_idx], corpus_idx, rng);
            // Ensure we generate non-zero voices
            while child_voices[corpus_idx].n_voices == 0 {
                child_voices[corpus_idx] = generate_corpus_voices(&corpuses[corpus_idx], corpus_idx, rng);
            }
        } else {
            // Randomly select one of the participating instruments to mutate
            let idx = *participating_indices.choose(rng).unwrap();
            child_voices[idx] = generate_corpus_voices(&corpuses[idx], idx, rng);
        }
    }

    let fitness = calculate_fitness(&child_voices, corpuses, target);
    Individual {
        entries: child_voices,
        fitness,
    }
}
/// Perform mutation on an individual
// fn mutate(
//     parent: &Individual,
//     corpuses: &[Corpus],
//     target: &[f32],
//     rng: &mut impl Rng
// ) -> Individual {
//     let mut child_voices = parent.entries.clone();
//     let corpus_idx = rng.random_range(0..corpuses.len());
//     let corpus = &corpuses[corpus_idx];

//     if rng.random_bool(0.3) {
//         // Change number of voices
//         child_voices[corpus_idx] = generate_corpus_voices(corpus, corpus_idx, rng);
//     } else {
//         // Change one random entry while maintaining uniqueness
//         let voice = &mut child_voices[corpus_idx];
//         if !voice.entry_indices.is_empty() {
//             let entry_idx = rng.random_range(0..voice.entry_indices.len());
//             let available: Vec<usize> = (0..corpus.entries.len())
//                 .filter(|&i| !voice.entry_indices.contains(&i))
//                 .collect();

//             if !available.is_empty() {
//                 let new_idx = available[rng.random_range(0..available.len())];

//                 if voice.n_voices == 2 &&
//                     (corpus.instrument == Instrument::Violin || corpus.instrument == Instrument::Cello) {
//                     // Need to validate double stop
//                     let other_idx = if entry_idx == 0 { 1 } else { 0 };
//                     let other_entry = &corpus.entries[voice.entry_indices[other_idx]];

//                     if can_play_double_stop(
//                         &corpus.entries[new_idx],
//                         other_entry,
//                         corpus.instrument
//                     ) {
//                         voice.entry_indices[entry_idx] = new_idx;
//                     }
//                 } else {
//                     voice.entry_indices[entry_idx] = new_idx;
//                 }
//             }
//         }
//     }

//     let fitness = calculate_fitness(&child_voices, corpuses, target);
//     Individual { entries: child_voices, fitness }
// }

pub fn genetic_search(
    target: &[f32],
    corpuses: &[Corpus],
    population_size: usize,
    beam_size: usize,
    n_iterations: usize,
) -> Vec<CorpusVoices> {
    let mut rng = rand::rng();
    let target_features = feature_vector(target);

    // Initialize population
    let mut population: Vec<Individual> = (0..population_size)
        .map(|_| {
            let voices: Vec<CorpusVoices> = corpuses
                .iter()
                .enumerate()
                .map(|(idx, corpus)| generate_corpus_voices(corpus, idx, &mut rng))
                .collect();

            let fitness = calculate_fitness(&voices, corpuses, &target_features);
            Individual {
                entries: voices,
                fitness,
            }
        })
        .collect();

    // Main evolution loop
    for iteration in 0..n_iterations {
        population.sort_by(|a, b| b.partial_cmp(a).unwrap());
        population.truncate(beam_size);

        println!(
            "Iteration {}: Best fitness = {}, Average fitness = {}",
            iteration,
            population[0].fitness,
            population.iter().map(|ind| ind.fitness).sum::<f32>() / population.len() as f32
        );

        let to_create = population_size - population.len();
        let mut new_individuals = Vec::with_capacity(to_create);

        while new_individuals.len() < to_create {
            if rng.random_bool(0.7) {
                let parent1 = population.choose(&mut rng).unwrap();
                let parent2 = population.choose(&mut rng).unwrap();
                new_individuals.push(crossover(parent1, parent2, corpuses, &target_features, &mut rng));
            } else {
                let parent = population.choose(&mut rng).unwrap();
                new_individuals.push(mutate(parent, corpuses, &target_features, &mut rng));
            }
        }

        population.extend(new_individuals);
    }

    population.sort_by(|a, b| b.partial_cmp(a).unwrap());
    population[0].entries.clone()
}
