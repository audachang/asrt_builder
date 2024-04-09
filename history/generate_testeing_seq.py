# Generate testing sequences
for block_index in range(number_of_learning_blocks):
    # Select pattern 1 for the learning phase
    selected_pattern = pattern_sequences[1]
    selected_pattern_answers = np.array(pattern_answers_indices[1])
    selected_pattern_answers_str = np.array(pattern_answers_directions[1])
    
    # Generate an initial random sequence for the first 5 trials
    initial_random_trials = np.random.randint(0, 4, 5)
    
    # Create ordered and random sequences of patterns, then interleave them
    ordered_pattern_sequence = np.tile(selected_pattern, repetitions_per_block)
    shuffled_pattern_sequence = ordered_pattern_sequence.copy()
    np.random.shuffle(shuffled_pattern_sequence)

    # Interleave the ordered and shuffled sequences, prepend the initial random trials
    learning_sequence = np.empty(len(ordered_pattern_sequence) + len(shuffled_pattern_sequence) + len(initial_random_trials), dtype=int)
    learning_sequence[:len(initial_random_trials)] = initial_random_trials
    learning_sequence[len(initial_random_trials)::2] = ordered_pattern_sequence
    learning_sequence[len(initial_random_trials)+1::2] = shuffled_pattern_sequence

    # Map sequence indices to answer directions
    learning_sequence_answers = selected_pattern_answers[learning_sequence]
    learning_sequence_answers_str = selected_pattern_answers_str[learning_sequence]

    # Save the learning sequence to a CSV file
    learning_sequence_df = pd.DataFrame({
        'orientation_degrees': learning_sequence * 90,
        'orientation_index': learning_sequence,
        'correct_answer_index': learning_sequence_answers,
        'correct_answer_direction': learning_sequence_answers_str
    })
    learning_filename = f'sequences/{participant_id}_{learning_condition_id}_learning_sequence_{block_index:02}.csv'
    learning_sequence_df.to_csv(learning_filename, index=False)
    learning_sequence_files.append(learning_filename)

# Save filenames of learning sequences to an Excel file
df_learning_files = pd.DataFrame({'learning_seq_files': learning_sequence_files})
df_learning_files.to_excel(f'sequences/learning_sequence_file_list.xlsx', index=False)

