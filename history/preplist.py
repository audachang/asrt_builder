import numpy as np
import pandas as pd

# Experiment setup: gather participant ID and learning condition type
participant_id = expInfo['participant']
learning_condition_id = expInfo['learning_type']

# Define experiment parameters
repetitions_per_block = 10
number_of_learning_blocks = 20
number_of_practice_blocks = 5
answer_directions = np.array(['up', 'right', 'down', 'left'])

# Define patterns and corresponding answers
# These mappings relate pattern IDs to sequences of correct answer indices
pattern_sequences = {1: [1, 2, 0, 3], 2: [2, 3, 1, 0]}
pattern_answers_indices = {1: [1, 2, 3, 0], 2: [1, 2, 3, 0]}
pattern_answers_directions = {1: ['right', 'down', 'left', 'up'],
                              2: ['right', 'down', 'left', 'up']}

# Lists to store filenames of generated sequence files
learning_sequence_files = []
practice_sequence_files = []

# Generate learning sequences
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

# Generate practice sequences
basic_orientation_indices = [0, 1, 2, 3]  # Basic orientations
practice_answers_indices = np.array([1, 2, 3, 0])
practice_answers_directions = np.array(['right', 'down', 'left', 'up'])

practice_sequence = np.tile(basic_orientation_indices, 21)
practice_sequence = np.append(practice_sequence, np.random.randint(0, 4))  # Ensure 85 trials in total

for block_index in range(number_of_practice_blocks):
    np.random.shuffle(practice_sequence)
    practice_sequence_answers = practice_answers_indices[practice_sequence]
    practice_sequence_answers_str = practice_answers_directions[practice_sequence]
    
    # Save the practice sequence to a CSV file
    practice_sequence_df = pd.DataFrame({
        'orientation_degrees': practice_sequence * 90,
        'orientation_index': practice_sequence,
        'correct_answer_index': practice_sequence_answers,
        'correct_answer_direction': practice_sequence_answers_str
    })    
    practice_filename = f'sequences/{participant_id}_{learning_condition_id}_practice_sequence_{block_index:02}.csv'
    practice_sequence_df.to_csv(practice_filename, index=False)
    practice_sequence_files.append(practice_filename)

# Save filenames of practice sequences to an Excel file
df_practice_files = pd.DataFrame({'practice_seq_files': practice_sequence_files})
df_practice_files.to_excel(f'sequences/practice_sequence_file_list.xlsx', index=False)

# Generate and save an initial random block of 85 trials
initial_random_sequence = np.random.randint(0, 4, 85)
initial_random_answers_directions = answer_directions[initial_random_sequence]
initial_random_sequence_df = pd.DataFrame({
    'orientation_index': initial_random_sequence,
    'orientation_degrees': initial_random_sequence * 90,
    'correct_answer_index': initial_random_sequence,
    'correct_answer_direction': initial_random_answers_directions
})
initial_random_sequence_df.to_csv(f'sequences/{participant_id}_{learning_condition_id}_initial_random.csv', index=False)

# Save the filename of the initial random sequence to an Excel file
df_initial_random_files = pd.DataFrame({'initial_random_seq_files': [f'sequences/{participant_id}_{learning_condition_id}_initial_random.csv']})
df_initial_random_files.to_excel('sequences/init_random_sequence_file_list.xlsx', index=True)
