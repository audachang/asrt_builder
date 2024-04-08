#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on April 08, 2024, at 21:03
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

import psychopy
psychopy.useVersion('2023.2.3')


# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'ASRT_motor'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'learning_type': 'motor',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\mypython\\Experiments\\ASRT_builder\\ASRT_motor_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1280, 800], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='mylaptop57cm', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='deg'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'deg'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcomescreen" ---
    spaceship = visual.ImageStim(
        win=win,
        name='spaceship', 
        image='images/spaceship.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(128, 64),
        color=[1,1,1], colorSpace='rgb', opacity=0.3,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    welcometext = visual.TextStim(win=win, name='welcometext',
        text='歡迎參加這個實驗！\n\n請您以上下左右四個按鍵操控一艘太空船，\n請依螢幕中央顯示箭號所指方向，\n按下相對應方向的按鍵\n\n準備好請按任意鍵開始…',
        font='DFKai-SB',
        pos=(0, 0), height=2.0, wrapWidth=30.0, ori=0.0, 
        color='cyan', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    welcome_end = keyboard.Keyboard()
    # Run 'Begin Experiment' code from preplist
    import numpy as np
    import pandas as pd
    
    # Experiment setup: gather participant ID and learning condition type
    participant_id = expInfo['participant']
    #learning_condition_id = expInfo['learning_type']
    
    # Define experiment parameters
    repetitions_per_block = 10
    number_of_learning_blocks = 20
    number_of_practice_blocks = 5
    number_of_motortesting_blocks = 5
    number_of_percepttesting_blocks = 5
    answer_directions = np.array(['up', 'right', 'down', 'left'])
    
    # Define patterns and corresponding answers
    # These mappings relate pattern IDs to sequences of correct answer indices
    pattern_sequences = {1: [1, 2, 0, 3], 2: [2, 3, 1, 0]}
    pattern_answers_indices = {1: [1, 2, 3, 0], 2: [1, 2, 3, 0]}
    pattern_answers_directions = {1: ['right', 'down', 'left', 'up'],
                                  2: ['up', 'right', 'down', 'left']}
    
    # Lists to store filenames of generated sequence files
    learning_sequence_files = []
    practice_sequence_files = []
    motortesting_sequence_files = []
    percepttesting_sequence_files = []
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
        learning_filename = f'sequences/{participant_id}_learning_sequence_{block_index:02}.csv'
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
        practice_filename = f'sequences/{participant_id}_practice_sequence_{block_index:02}.csv'
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
    initial_random_sequence_df.to_csv(f'sequences/{participant_id}_initial_random.csv', index=False)
    
    # Save the filename of the initial random sequence to an Excel file
    df_initial_random_files = pd.DataFrame({'initial_random_seq_files': [f'sequences/{participant_id}_initial_random.csv']})
    df_initial_random_files.to_excel('sequences/init_random_sequence_file_list.xlsx', index=True)
    
    # Generate motor testing sequences
    for block_index in range(number_of_motortesting_blocks):
        # Select pattern 2 for the testing phase
        selected_pattern = pattern_sequences[2]
        selected_pattern_answers = np.array(pattern_answers_indices[2])
        selected_pattern_answers_str = np.array(pattern_answers_directions[2])
        
        # Generate an initial random sequence for the first 5 trials
        initial_random_trials = np.random.randint(0, 4, 5)
        
        # Create ordered and random sequences of patterns, then interleave them
        ordered_pattern_sequence = np.tile(selected_pattern, repetitions_per_block)
        shuffled_pattern_sequence = ordered_pattern_sequence.copy()
        np.random.shuffle(shuffled_pattern_sequence)
    
        # Interleave the ordered and shuffled sequences, prepend the initial random trials
        motortesting_sequence = np.empty(len(ordered_pattern_sequence) + len(shuffled_pattern_sequence) + len(initial_random_trials), dtype=int)
        motortesting_sequence[:len(initial_random_trials)] = initial_random_trials
        motortesting_sequence[len(initial_random_trials)::2] = ordered_pattern_sequence
        motortesting_sequence[len(initial_random_trials)+1::2] = shuffled_pattern_sequence
    
        # Map sequence indices to answer directions
        motortesting_sequence_answers = selected_pattern_answers[motortesting_sequence]
        motortesting_sequence_answers_str = selected_pattern_answers_str[motortesting_sequence]
    
        # Save the testing sequence to a CSV file
        motortesting_sequence_df = pd.DataFrame({
            'orientation_degrees': motortesting_sequence * 90,
            'orientation_index': motortesting_sequence,
            'correct_answer_index': motortesting_sequence_answers,
            'correct_answer_direction': motortesting_sequence_answers_str
        })
        motortesting_filename = f'sequences/{participant_id}_motortesting_sequence_{block_index:02}.csv'
        motortesting_sequence_df.to_csv(motortesting_filename, index=False)
        motortesting_sequence_files.append(motortesting_filename)
    
    # Save filenames of testing sequences to an Excel file
    df_motortesting_files = pd.DataFrame({'motor_testing_seq_files': motortesting_sequence_files})
    df_motortesting_files.to_excel(f'sequences/motor_testing_sequence_file_list.xlsx', index=False)
    
    # Generate perceptual testing sequences
    for block_index in range(number_of_percepttesting_blocks):
        # Select pattern 1 for the testing phase
        selected_pattern = pattern_sequences[1]
        selected_pattern_answers = np.array(pattern_answers_indices[2])
        selected_pattern_answers_str = np.array(pattern_answers_directions[2])
        
        # Generate an initial random sequence for the first 5 trials
        initial_random_trials = np.random.randint(0, 4, 5)
        
        # Create ordered and random sequences of patterns, then interleave them
        ordered_pattern_sequence = np.tile(selected_pattern, repetitions_per_block)
        shuffled_pattern_sequence = ordered_pattern_sequence.copy()
        np.random.shuffle(shuffled_pattern_sequence)
    
        # Interleave the ordered and shuffled sequences, prepend the initial random trials
        percepttesting_sequence = np.empty(len(ordered_pattern_sequence) + len(shuffled_pattern_sequence) + len(initial_random_trials), dtype=int)
        percepttesting_sequence[:len(initial_random_trials)] = initial_random_trials
        percepttesting_sequence[len(initial_random_trials)::2] = ordered_pattern_sequence
        percepttesting_sequence[len(initial_random_trials)+1::2] = shuffled_pattern_sequence
    
        # Map sequence indices to answer directions
        percepttesting_sequence_answers = selected_pattern_answers[percepttesting_sequence]
        percepttesting_sequence_answers_str = selected_pattern_answers_str[percepttesting_sequence]
    
        # Save the testing sequence to a CSV file
        percepttesting_sequence_df = pd.DataFrame({
            'orientation_degrees': percepttesting_sequence * 90,
            'orientation_index': percepttesting_sequence,
            'correct_answer_index': percepttesting_sequence_answers,
            'correct_answer_direction': percepttesting_sequence_answers_str
        })
        percepttesting_filename = f'sequences/{participant_id}_percepttesting_sequence_{block_index:02}.csv'
        percepttesting_sequence_df.to_csv(percepttesting_filename, index=False)
        percepttesting_sequence_files.append(percepttesting_filename)
    
    # Save filenames of testing sequences to an Excel file
    df_percepttesting_files = pd.DataFrame({'percept_testing_seq_files': percepttesting_sequence_files})
    df_percepttesting_files.to_excel(f'sequences/percept_testing_sequence_file_list.xlsx', index=False)
    
    # --- Initialize components for Routine "pre_section" ---
    
    # --- Initialize components for Routine "trial" ---
    arrowhead = visual.ShapeStim(
        win=win, name='arrowhead', vertices='arrow',units='deg', 
        size=(3,3),
        ori=1.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "fixation_practice" ---
    cross = visual.ShapeStim(
        win=win, name='cross', vertices='cross',
        size=(2, 2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    # Run 'Begin Experiment' code from feedback_code
    feedbackstr=""
    feedback_text = visual.TextStim(win=win, name='feedback_text',
        text='',
        font='DFKai-SB',
        pos=(0, -5), height=3.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "post_section" ---
    feedback_post_section = visual.TextStim(win=win, name='feedback_post_section',
        text='',
        font='DFKai-SB',
        pos=(0, 0), height=2.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_4 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "pratice_instruction" ---
    spaceship_rotated = visual.ImageStim(
        win=win,
        name='spaceship_rotated', 
        image='images/spaceship.jpg', mask=None, anchor='center',
        ori=-45.0, pos=(0, 0), size=(128, 64),
        color=[1,1,1], colorSpace='rgb', opacity=0.3,
        flipHoriz=False, flipVert=True,
        texRes=128.0, interpolate=True, depth=0.0)
    rotation_instruction = visual.TextStim(win=win, name='rotation_instruction',
        text='現在因為方向控制系統故障，\n太空船產生逆時鐘90度的方向偏誤，\n您必須選擇顯示箭號順時鐘旋轉90度後之按鍵：\n\n看到↑，請按→\n看到→，請按↓\n看到↓，請按←\n看到←，請按↑\n\n準備好請按任意鍵繼續…\n\n\n\n\n',
        font='DFKai-SB',
        pos=(0, 0), height=2.0, wrapWidth=30.0, ori=0.0, 
        color='cyan', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "pre_section" ---
    
    # --- Initialize components for Routine "trial" ---
    arrowhead = visual.ShapeStim(
        win=win, name='arrowhead', vertices='arrow',units='deg', 
        size=(3,3),
        ori=1.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "fixation_practice" ---
    cross = visual.ShapeStim(
        win=win, name='cross', vertices='cross',
        size=(2, 2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    # Run 'Begin Experiment' code from feedback_code
    feedbackstr=""
    feedback_text = visual.TextStim(win=win, name='feedback_text',
        text='',
        font='DFKai-SB',
        pos=(0, -5), height=3.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "post_section" ---
    feedback_post_section = visual.TextStim(win=win, name='feedback_post_section',
        text='',
        font='DFKai-SB',
        pos=(0, 0), height=2.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_4 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "pre_section" ---
    
    # --- Initialize components for Routine "trial" ---
    arrowhead = visual.ShapeStim(
        win=win, name='arrowhead', vertices='arrow',units='deg', 
        size=(3,3),
        ori=1.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "fixation" ---
    cross_2 = visual.ShapeStim(
        win=win, name='cross_2', vertices='cross',
        size=(2, 2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "post_section" ---
    feedback_post_section = visual.TextStim(win=win, name='feedback_post_section',
        text='',
        font='DFKai-SB',
        pos=(0, 0), height=2.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_4 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "motor_testing_instruction" ---
    spaceship3 = visual.ImageStim(
        win=win,
        name='spaceship3', 
        image='images/spaceship.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(128, 64),
        color=[1,1,1], colorSpace='rgb', opacity=0.3,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    motor_testing_instruction_text = visual.TextStim(win=win, name='motor_testing_instruction_text',
        text='現在太空船已維修好正常運作，\n請您依箭頭所指方向按下反應鍵，\n不要再選擇旋轉後的方向。\n\n準備好請按任意鍵繼續…',
        font='DFKai-SB',
        pos=(0, 0), height=2.0, wrapWidth=30.0, ori=0.0, 
        color='cyan', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_3 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "pre_section" ---
    
    # --- Initialize components for Routine "trial" ---
    arrowhead = visual.ShapeStim(
        win=win, name='arrowhead', vertices='arrow',units='deg', 
        size=(3,3),
        ori=1.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "fixation" ---
    cross_2 = visual.ShapeStim(
        win=win, name='cross_2', vertices='cross',
        size=(2, 2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "post_section" ---
    feedback_post_section = visual.TextStim(win=win, name='feedback_post_section',
        text='',
        font='DFKai-SB',
        pos=(0, 0), height=2.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_4 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "percept_testing_instruction" ---
    spaceship4 = visual.ImageStim(
        win=win,
        name='spaceship4', 
        image='images/spaceship.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(128, 64),
        color=[1,1,1], colorSpace='rgb', opacity=0.3,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_5 = keyboard.Keyboard()
    percept_testing_instruction_text = visual.TextStim(win=win, name='percept_testing_instruction_text',
        text='請您繼續依箭頭所指方向按下反應鍵。\n準備好請按任意鍵繼續…',
        font='DFKai-SB',
        pos=(0, 0), height=2.0, wrapWidth=30.0, ori=0.0, 
        color='cyan', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "pre_section" ---
    
    # --- Initialize components for Routine "trial" ---
    arrowhead = visual.ShapeStim(
        win=win, name='arrowhead', vertices='arrow',units='deg', 
        size=(3,3),
        ori=1.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "fixation" ---
    cross_2 = visual.ShapeStim(
        win=win, name='cross_2', vertices='cross',
        size=(2, 2),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "post_section" ---
    feedback_post_section = visual.TextStim(win=win, name='feedback_post_section',
        text='',
        font='DFKai-SB',
        pos=(0, 0), height=2.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_4 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "thankyouscreen" ---
    text = visual.TextStim(win=win, name='text',
        text='實驗已完成，謝謝您的參與！\n請等待實驗者的進一步導引。',
        font='DFKai-SB',
        pos=(0, 0), height=3.0, wrapWidth=30.0, ori=0.0, 
        color='cyan', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_6 = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "welcomescreen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('welcomescreen.started', globalClock.getTime())
    welcome_end.keys = []
    welcome_end.rt = []
    _welcome_end_allKeys = []
    # Run 'Begin Routine' code from preplist
    bkNum = 0
    bkRT = []
    bkCorrectNum = 0
    bkTrialNum = 0
    
    # keep track of which components have finished
    welcomescreenComponents = [spaceship, welcometext, welcome_end]
    for thisComponent in welcomescreenComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcomescreen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *spaceship* updates
        
        # if spaceship is starting this frame...
        if spaceship.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            spaceship.frameNStart = frameN  # exact frame index
            spaceship.tStart = t  # local t and not account for scr refresh
            spaceship.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(spaceship, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'spaceship.started')
            # update status
            spaceship.status = STARTED
            spaceship.setAutoDraw(True)
        
        # if spaceship is active this frame...
        if spaceship.status == STARTED:
            # update params
            pass
        
        # *welcometext* updates
        
        # if welcometext is starting this frame...
        if welcometext.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcometext.frameNStart = frameN  # exact frame index
            welcometext.tStart = t  # local t and not account for scr refresh
            welcometext.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcometext, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcometext.started')
            # update status
            welcometext.status = STARTED
            welcometext.setAutoDraw(True)
        
        # if welcometext is active this frame...
        if welcometext.status == STARTED:
            # update params
            pass
        
        # *welcome_end* updates
        waitOnFlip = False
        
        # if welcome_end is starting this frame...
        if welcome_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_end.frameNStart = frameN  # exact frame index
            welcome_end.tStart = t  # local t and not account for scr refresh
            welcome_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_end.started')
            # update status
            welcome_end.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(welcome_end.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(welcome_end.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if welcome_end.status == STARTED and not waitOnFlip:
            theseKeys = welcome_end.getKeys(keyList=['space', 'up','left', 'down', 'right'], ignoreKeys=["escape"], waitRelease=False)
            _welcome_end_allKeys.extend(theseKeys)
            if len(_welcome_end_allKeys):
                welcome_end.keys = _welcome_end_allKeys[-1].name  # just the last key pressed
                welcome_end.rt = _welcome_end_allKeys[-1].rt
                welcome_end.duration = _welcome_end_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcomescreenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcomescreen" ---
    for thisComponent in welcomescreenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('welcomescreen.stopped', globalClock.getTime())
    # check responses
    if welcome_end.keys in ['', [], None]:  # No response was made
        welcome_end.keys = None
    thisExp.addData('welcome_end.keys',welcome_end.keys)
    if welcome_end.keys != None:  # we had a response
        thisExp.addData('welcome_end.rt', welcome_end.rt)
        thisExp.addData('welcome_end.duration', welcome_end.duration)
    thisExp.nextEntry()
    # the Routine "welcomescreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    init_random_block = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('sequences/init_random_sequence_file_list.xlsx'),
        seed=None, name='init_random_block')
    thisExp.addLoop(init_random_block)  # add the loop to the experiment
    thisInit_random_block = init_random_block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisInit_random_block.rgb)
    if thisInit_random_block != None:
        for paramName in thisInit_random_block:
            globals()[paramName] = thisInit_random_block[paramName]
    
    for thisInit_random_block in init_random_block:
        currentLoop = init_random_block
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisInit_random_block.rgb)
        if thisInit_random_block != None:
            for paramName in thisInit_random_block:
                globals()[paramName] = thisInit_random_block[paramName]
        
        # --- Prepare to start Routine "pre_section" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('pre_section.started', globalClock.getTime())
        # Run 'Begin Routine' code from performance_pre_section
        bkNum += 1
        # keep track of which components have finished
        pre_sectionComponents = []
        for thisComponent in pre_sectionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pre_section" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pre_sectionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pre_section" ---
        for thisComponent in pre_sectionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('pre_section.stopped', globalClock.getTime())
        # the Routine "pre_section" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        init_random_trials = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions(initial_random_seq_files, selection='0:10'),
            seed=None, name='init_random_trials')
        thisExp.addLoop(init_random_trials)  # add the loop to the experiment
        thisInit_random_trial = init_random_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisInit_random_trial.rgb)
        if thisInit_random_trial != None:
            for paramName in thisInit_random_trial:
                globals()[paramName] = thisInit_random_trial[paramName]
        
        for thisInit_random_trial in init_random_trials:
            currentLoop = init_random_trials
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisInit_random_trial.rgb)
            if thisInit_random_trial != None:
                for paramName in thisInit_random_trial:
                    globals()[paramName] = thisInit_random_trial[paramName]
            
            # --- Prepare to start Routine "trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial.started', globalClock.getTime())
            arrowhead.setOri(orientation_degrees)
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            # keep track of which components have finished
            trialComponents = [arrowhead, key_resp]
            for thisComponent in trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *arrowhead* updates
                
                # if arrowhead is starting this frame...
                if arrowhead.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    arrowhead.frameNStart = frameN  # exact frame index
                    arrowhead.tStart = t  # local t and not account for scr refresh
                    arrowhead.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(arrowhead, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'arrowhead.started')
                    # update status
                    arrowhead.status = STARTED
                    arrowhead.setAutoDraw(True)
                
                # if arrowhead is active this frame...
                if arrowhead.status == STARTED:
                    # update params
                    pass
                
                # *key_resp* updates
                waitOnFlip = False
                
                # if key_resp is starting this frame...
                if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.started')
                    # update status
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['left','right','up','down'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                        key_resp.rt = _key_resp_allKeys[0].rt
                        key_resp.duration = _key_resp_allKeys[0].duration
                        # was this correct?
                        if (key_resp.keys == str(correct_answer_direction)) or (key_resp.keys == correct_answer_direction):
                            key_resp.corr = 1
                        else:
                            key_resp.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial.stopped', globalClock.getTime())
            # check responses
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
                # was no response the correct answer?!
                if str(correct_answer_direction).lower() == 'none':
                   key_resp.corr = 1;  # correct non-response
                else:
                   key_resp.corr = 0;  # failed to respond (incorrectly)
            # store data for init_random_trials (TrialHandler)
            init_random_trials.addData('key_resp.keys',key_resp.keys)
            init_random_trials.addData('key_resp.corr', key_resp.corr)
            if key_resp.keys != None:  # we had a response
                init_random_trials.addData('key_resp.rt', key_resp.rt)
                init_random_trials.addData('key_resp.duration', key_resp.duration)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "fixation_practice" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('fixation_practice.started', globalClock.getTime())
            # Run 'Begin Routine' code from feedback_code
            if key_resp.keys == correct_answer_direction:
                feedbackstr ="正確"
                bkCorrectNum += 1
            else:
                feedbackstr = "錯誤"
                
            
            bkRT.append(key_resp.rt)
            bkTrialNum+=1
            feedback_text.setText(feedbackstr)
            # keep track of which components have finished
            fixation_practiceComponents = [cross, feedback_text]
            for thisComponent in fixation_practiceComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation_practice" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *cross* updates
                
                # if cross is starting this frame...
                if cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cross.frameNStart = frameN  # exact frame index
                    cross.tStart = t  # local t and not account for scr refresh
                    cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross.started')
                    # update status
                    cross.status = STARTED
                    cross.setAutoDraw(True)
                
                # if cross is active this frame...
                if cross.status == STARTED:
                    # update params
                    pass
                
                # if cross is stopping this frame...
                if cross.status == STARTED:
                    if frameN >= 30:
                        # keep track of stop time/frame for later
                        cross.tStop = t  # not accounting for scr refresh
                        cross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cross.stopped')
                        # update status
                        cross.status = FINISHED
                        cross.setAutoDraw(False)
                
                # *feedback_text* updates
                
                # if feedback_text is starting this frame...
                if feedback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    feedback_text.frameNStart = frameN  # exact frame index
                    feedback_text.tStart = t  # local t and not account for scr refresh
                    feedback_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(feedback_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_text.started')
                    # update status
                    feedback_text.status = STARTED
                    feedback_text.setAutoDraw(True)
                
                # if feedback_text is active this frame...
                if feedback_text.status == STARTED:
                    # update params
                    pass
                
                # if feedback_text is stopping this frame...
                if feedback_text.status == STARTED:
                    if frameN >= 60:
                        # keep track of stop time/frame for later
                        feedback_text.tStop = t  # not accounting for scr refresh
                        feedback_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'feedback_text.stopped')
                        # update status
                        feedback_text.status = FINISHED
                        feedback_text.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixation_practiceComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation_practice" ---
            for thisComponent in fixation_practiceComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('fixation_practice.stopped', globalClock.getTime())
            # the Routine "fixation_practice" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'init_random_trials'
        
        
        # --- Prepare to start Routine "post_section" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('post_section.started', globalClock.getTime())
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not ((bkNum < 5 ) and (bkNum % 5 != 0 ) )
        # Run 'Begin Routine' code from performance_post_section
        if len(bkRT) < 1:
            bkRT.append(0)
        if bkTrialNum < 1:
            bkTrialNum = 1
            
        bkRTmean = int(np.mean(bkRT)*1000)
        bkACC = bkCorrectNum / bkTrialNum * 100
        bkRTmeanstr = f"平均反應時間：{bkRTmean} ms"
        bkACCstr = f"正確率：{bkACC}%"
        sentence1 = "您在這一個段落的表現："
        sentence2 = "請按任意鍵繼續…"
        this_feedbacktext = f"""
        {sentence1}({bkTrialNum} trials, {bkNum} blocks)
        {bkRTmeanstr}
        {bkACCstr}
        {sentence2}
        """
        feedback_post_section.setText(this_feedbacktext)
        key_resp_4.keys = []
        key_resp_4.rt = []
        _key_resp_4_allKeys = []
        # keep track of which components have finished
        post_sectionComponents = [feedback_post_section, key_resp_4]
        for thisComponent in post_sectionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "post_section" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_post_section* updates
            
            # if feedback_post_section is starting this frame...
            if feedback_post_section.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_post_section.frameNStart = frameN  # exact frame index
                feedback_post_section.tStart = t  # local t and not account for scr refresh
                feedback_post_section.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_post_section, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_post_section.started')
                # update status
                feedback_post_section.status = STARTED
                feedback_post_section.setAutoDraw(True)
            
            # if feedback_post_section is active this frame...
            if feedback_post_section.status == STARTED:
                # update params
                pass
            
            # *key_resp_4* updates
            waitOnFlip = False
            
            # if key_resp_4 is starting this frame...
            if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_4.frameNStart = frameN  # exact frame index
                key_resp_4.tStart = t  # local t and not account for scr refresh
                key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_4.started')
                # update status
                key_resp_4.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_4.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_4.getKeys(keyList=['space', 'up','left', 'down', 'right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_4_allKeys.extend(theseKeys)
                if len(_key_resp_4_allKeys):
                    key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                    key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                    key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in post_sectionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "post_section" ---
        for thisComponent in post_sectionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('post_section.stopped', globalClock.getTime())
        # check responses
        if key_resp_4.keys in ['', [], None]:  # No response was made
            key_resp_4.keys = None
        init_random_block.addData('key_resp_4.keys',key_resp_4.keys)
        if key_resp_4.keys != None:  # we had a response
            init_random_block.addData('key_resp_4.rt', key_resp_4.rt)
            init_random_block.addData('key_resp_4.duration', key_resp_4.duration)
        # the Routine "post_section" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 1.0 repeats of 'init_random_block'
    
    
    # --- Prepare to start Routine "pratice_instruction" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('pratice_instruction.started', globalClock.getTime())
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # Run 'Begin Routine' code from pre_prac
    bkNum = 0
    bkRT = []
    bkCorrectNum = 0
    bkTrialNum = 0
    
    # keep track of which components have finished
    pratice_instructionComponents = [spaceship_rotated, rotation_instruction, key_resp_2]
    for thisComponent in pratice_instructionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pratice_instruction" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *spaceship_rotated* updates
        
        # if spaceship_rotated is starting this frame...
        if spaceship_rotated.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            spaceship_rotated.frameNStart = frameN  # exact frame index
            spaceship_rotated.tStart = t  # local t and not account for scr refresh
            spaceship_rotated.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(spaceship_rotated, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'spaceship_rotated.started')
            # update status
            spaceship_rotated.status = STARTED
            spaceship_rotated.setAutoDraw(True)
        
        # if spaceship_rotated is active this frame...
        if spaceship_rotated.status == STARTED:
            # update params
            pass
        
        # *rotation_instruction* updates
        
        # if rotation_instruction is starting this frame...
        if rotation_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rotation_instruction.frameNStart = frameN  # exact frame index
            rotation_instruction.tStart = t  # local t and not account for scr refresh
            rotation_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rotation_instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rotation_instruction.started')
            # update status
            rotation_instruction.status = STARTED
            rotation_instruction.setAutoDraw(True)
        
        # if rotation_instruction is active this frame...
        if rotation_instruction.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['space', 'up','left', 'down', 'right'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pratice_instructionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pratice_instruction" ---
    for thisComponent in pratice_instructionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('pratice_instruction.stopped', globalClock.getTime())
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "pratice_instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_blocks = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('sequences/practice_sequence_file_list.xlsx'),
        seed=None, name='practice_blocks')
    thisExp.addLoop(practice_blocks)  # add the loop to the experiment
    thisPractice_block = practice_blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_block.rgb)
    if thisPractice_block != None:
        for paramName in thisPractice_block:
            globals()[paramName] = thisPractice_block[paramName]
    
    for thisPractice_block in practice_blocks:
        currentLoop = practice_blocks
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_block.rgb)
        if thisPractice_block != None:
            for paramName in thisPractice_block:
                globals()[paramName] = thisPractice_block[paramName]
        
        # --- Prepare to start Routine "pre_section" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('pre_section.started', globalClock.getTime())
        # Run 'Begin Routine' code from performance_pre_section
        bkNum += 1
        # keep track of which components have finished
        pre_sectionComponents = []
        for thisComponent in pre_sectionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pre_section" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pre_sectionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pre_section" ---
        for thisComponent in pre_sectionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('pre_section.stopped', globalClock.getTime())
        # the Routine "pre_section" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        practice_trials = data.TrialHandler(nReps=1.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions(practice_seq_files, selection='0:10'),
            seed=None, name='practice_trials')
        thisExp.addLoop(practice_trials)  # add the loop to the experiment
        thisPractice_trial = practice_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
        if thisPractice_trial != None:
            for paramName in thisPractice_trial:
                globals()[paramName] = thisPractice_trial[paramName]
        
        for thisPractice_trial in practice_trials:
            currentLoop = practice_trials
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
            if thisPractice_trial != None:
                for paramName in thisPractice_trial:
                    globals()[paramName] = thisPractice_trial[paramName]
            
            # --- Prepare to start Routine "trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial.started', globalClock.getTime())
            arrowhead.setOri(orientation_degrees)
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            # keep track of which components have finished
            trialComponents = [arrowhead, key_resp]
            for thisComponent in trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *arrowhead* updates
                
                # if arrowhead is starting this frame...
                if arrowhead.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    arrowhead.frameNStart = frameN  # exact frame index
                    arrowhead.tStart = t  # local t and not account for scr refresh
                    arrowhead.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(arrowhead, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'arrowhead.started')
                    # update status
                    arrowhead.status = STARTED
                    arrowhead.setAutoDraw(True)
                
                # if arrowhead is active this frame...
                if arrowhead.status == STARTED:
                    # update params
                    pass
                
                # *key_resp* updates
                waitOnFlip = False
                
                # if key_resp is starting this frame...
                if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.started')
                    # update status
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['left','right','up','down'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                        key_resp.rt = _key_resp_allKeys[0].rt
                        key_resp.duration = _key_resp_allKeys[0].duration
                        # was this correct?
                        if (key_resp.keys == str(correct_answer_direction)) or (key_resp.keys == correct_answer_direction):
                            key_resp.corr = 1
                        else:
                            key_resp.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial.stopped', globalClock.getTime())
            # check responses
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
                # was no response the correct answer?!
                if str(correct_answer_direction).lower() == 'none':
                   key_resp.corr = 1;  # correct non-response
                else:
                   key_resp.corr = 0;  # failed to respond (incorrectly)
            # store data for practice_trials (TrialHandler)
            practice_trials.addData('key_resp.keys',key_resp.keys)
            practice_trials.addData('key_resp.corr', key_resp.corr)
            if key_resp.keys != None:  # we had a response
                practice_trials.addData('key_resp.rt', key_resp.rt)
                practice_trials.addData('key_resp.duration', key_resp.duration)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "fixation_practice" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('fixation_practice.started', globalClock.getTime())
            # Run 'Begin Routine' code from feedback_code
            if key_resp.keys == correct_answer_direction:
                feedbackstr ="正確"
                bkCorrectNum += 1
            else:
                feedbackstr = "錯誤"
                
            
            bkRT.append(key_resp.rt)
            bkTrialNum+=1
            feedback_text.setText(feedbackstr)
            # keep track of which components have finished
            fixation_practiceComponents = [cross, feedback_text]
            for thisComponent in fixation_practiceComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation_practice" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *cross* updates
                
                # if cross is starting this frame...
                if cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cross.frameNStart = frameN  # exact frame index
                    cross.tStart = t  # local t and not account for scr refresh
                    cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross.started')
                    # update status
                    cross.status = STARTED
                    cross.setAutoDraw(True)
                
                # if cross is active this frame...
                if cross.status == STARTED:
                    # update params
                    pass
                
                # if cross is stopping this frame...
                if cross.status == STARTED:
                    if frameN >= 30:
                        # keep track of stop time/frame for later
                        cross.tStop = t  # not accounting for scr refresh
                        cross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cross.stopped')
                        # update status
                        cross.status = FINISHED
                        cross.setAutoDraw(False)
                
                # *feedback_text* updates
                
                # if feedback_text is starting this frame...
                if feedback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    feedback_text.frameNStart = frameN  # exact frame index
                    feedback_text.tStart = t  # local t and not account for scr refresh
                    feedback_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(feedback_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_text.started')
                    # update status
                    feedback_text.status = STARTED
                    feedback_text.setAutoDraw(True)
                
                # if feedback_text is active this frame...
                if feedback_text.status == STARTED:
                    # update params
                    pass
                
                # if feedback_text is stopping this frame...
                if feedback_text.status == STARTED:
                    if frameN >= 60:
                        # keep track of stop time/frame for later
                        feedback_text.tStop = t  # not accounting for scr refresh
                        feedback_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'feedback_text.stopped')
                        # update status
                        feedback_text.status = FINISHED
                        feedback_text.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixation_practiceComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation_practice" ---
            for thisComponent in fixation_practiceComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('fixation_practice.stopped', globalClock.getTime())
            # the Routine "fixation_practice" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'practice_trials'
        
        
        # --- Prepare to start Routine "post_section" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('post_section.started', globalClock.getTime())
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not ((bkNum < 5 ) and (bkNum % 5 != 0 ) )
        # Run 'Begin Routine' code from performance_post_section
        if len(bkRT) < 1:
            bkRT.append(0)
        if bkTrialNum < 1:
            bkTrialNum = 1
            
        bkRTmean = int(np.mean(bkRT)*1000)
        bkACC = bkCorrectNum / bkTrialNum * 100
        bkRTmeanstr = f"平均反應時間：{bkRTmean} ms"
        bkACCstr = f"正確率：{bkACC}%"
        sentence1 = "您在這一個段落的表現："
        sentence2 = "請按任意鍵繼續…"
        this_feedbacktext = f"""
        {sentence1}({bkTrialNum} trials, {bkNum} blocks)
        {bkRTmeanstr}
        {bkACCstr}
        {sentence2}
        """
        feedback_post_section.setText(this_feedbacktext)
        key_resp_4.keys = []
        key_resp_4.rt = []
        _key_resp_4_allKeys = []
        # keep track of which components have finished
        post_sectionComponents = [feedback_post_section, key_resp_4]
        for thisComponent in post_sectionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "post_section" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_post_section* updates
            
            # if feedback_post_section is starting this frame...
            if feedback_post_section.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_post_section.frameNStart = frameN  # exact frame index
                feedback_post_section.tStart = t  # local t and not account for scr refresh
                feedback_post_section.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_post_section, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_post_section.started')
                # update status
                feedback_post_section.status = STARTED
                feedback_post_section.setAutoDraw(True)
            
            # if feedback_post_section is active this frame...
            if feedback_post_section.status == STARTED:
                # update params
                pass
            
            # *key_resp_4* updates
            waitOnFlip = False
            
            # if key_resp_4 is starting this frame...
            if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_4.frameNStart = frameN  # exact frame index
                key_resp_4.tStart = t  # local t and not account for scr refresh
                key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_4.started')
                # update status
                key_resp_4.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_4.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_4.getKeys(keyList=['space', 'up','left', 'down', 'right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_4_allKeys.extend(theseKeys)
                if len(_key_resp_4_allKeys):
                    key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                    key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                    key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in post_sectionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "post_section" ---
        for thisComponent in post_sectionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('post_section.stopped', globalClock.getTime())
        # check responses
        if key_resp_4.keys in ['', [], None]:  # No response was made
            key_resp_4.keys = None
        practice_blocks.addData('key_resp_4.keys',key_resp_4.keys)
        if key_resp_4.keys != None:  # we had a response
            practice_blocks.addData('key_resp_4.rt', key_resp_4.rt)
            practice_blocks.addData('key_resp_4.duration', key_resp_4.duration)
        # the Routine "post_section" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'practice_blocks'
    
    
    # set up handler to look after randomisation of conditions etc
    learning_trials = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('sequences/learning_sequence_file_list.xlsx'),
        seed=None, name='learning_trials')
    thisExp.addLoop(learning_trials)  # add the loop to the experiment
    thisLearning_trial = learning_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLearning_trial.rgb)
    if thisLearning_trial != None:
        for paramName in thisLearning_trial:
            globals()[paramName] = thisLearning_trial[paramName]
    
    for thisLearning_trial in learning_trials:
        currentLoop = learning_trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisLearning_trial.rgb)
        if thisLearning_trial != None:
            for paramName in thisLearning_trial:
                globals()[paramName] = thisLearning_trial[paramName]
        
        # --- Prepare to start Routine "pre_section" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('pre_section.started', globalClock.getTime())
        # Run 'Begin Routine' code from performance_pre_section
        bkNum += 1
        # keep track of which components have finished
        pre_sectionComponents = []
        for thisComponent in pre_sectionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pre_section" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pre_sectionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pre_section" ---
        for thisComponent in pre_sectionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('pre_section.stopped', globalClock.getTime())
        # the Routine "pre_section" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        learning_loop = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions(learning_seq_files, selection='0:10'),
            seed=None, name='learning_loop')
        thisExp.addLoop(learning_loop)  # add the loop to the experiment
        thisLearning_loop = learning_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLearning_loop.rgb)
        if thisLearning_loop != None:
            for paramName in thisLearning_loop:
                globals()[paramName] = thisLearning_loop[paramName]
        
        for thisLearning_loop in learning_loop:
            currentLoop = learning_loop
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisLearning_loop.rgb)
            if thisLearning_loop != None:
                for paramName in thisLearning_loop:
                    globals()[paramName] = thisLearning_loop[paramName]
            
            # --- Prepare to start Routine "trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial.started', globalClock.getTime())
            arrowhead.setOri(orientation_degrees)
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            # keep track of which components have finished
            trialComponents = [arrowhead, key_resp]
            for thisComponent in trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *arrowhead* updates
                
                # if arrowhead is starting this frame...
                if arrowhead.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    arrowhead.frameNStart = frameN  # exact frame index
                    arrowhead.tStart = t  # local t and not account for scr refresh
                    arrowhead.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(arrowhead, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'arrowhead.started')
                    # update status
                    arrowhead.status = STARTED
                    arrowhead.setAutoDraw(True)
                
                # if arrowhead is active this frame...
                if arrowhead.status == STARTED:
                    # update params
                    pass
                
                # *key_resp* updates
                waitOnFlip = False
                
                # if key_resp is starting this frame...
                if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.started')
                    # update status
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['left','right','up','down'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                        key_resp.rt = _key_resp_allKeys[0].rt
                        key_resp.duration = _key_resp_allKeys[0].duration
                        # was this correct?
                        if (key_resp.keys == str(correct_answer_direction)) or (key_resp.keys == correct_answer_direction):
                            key_resp.corr = 1
                        else:
                            key_resp.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial.stopped', globalClock.getTime())
            # check responses
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
                # was no response the correct answer?!
                if str(correct_answer_direction).lower() == 'none':
                   key_resp.corr = 1;  # correct non-response
                else:
                   key_resp.corr = 0;  # failed to respond (incorrectly)
            # store data for learning_loop (TrialHandler)
            learning_loop.addData('key_resp.keys',key_resp.keys)
            learning_loop.addData('key_resp.corr', key_resp.corr)
            if key_resp.keys != None:  # we had a response
                learning_loop.addData('key_resp.rt', key_resp.rt)
                learning_loop.addData('key_resp.duration', key_resp.duration)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "fixation" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('fixation.started', globalClock.getTime())
            # Run 'Begin Routine' code from code
            if key_resp.keys == correct_answer_direction:
                feedbackstr ="正確"
                bkCorrectNum += 1
            else:
                feedbackstr = "錯誤"
                
            bkRT.append(key_resp.rt)
            bkTrialNum+=1
            # keep track of which components have finished
            fixationComponents = [cross_2]
            for thisComponent in fixationComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *cross_2* updates
                
                # if cross_2 is starting this frame...
                if cross_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cross_2.frameNStart = frameN  # exact frame index
                    cross_2.tStart = t  # local t and not account for scr refresh
                    cross_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cross_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_2.started')
                    # update status
                    cross_2.status = STARTED
                    cross_2.setAutoDraw(True)
                
                # if cross_2 is active this frame...
                if cross_2.status == STARTED:
                    # update params
                    pass
                
                # if cross_2 is stopping this frame...
                if cross_2.status == STARTED:
                    if frameN >= 8:
                        # keep track of stop time/frame for later
                        cross_2.tStop = t  # not accounting for scr refresh
                        cross_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cross_2.stopped')
                        # update status
                        cross_2.status = FINISHED
                        cross_2.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixationComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation" ---
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('fixation.stopped', globalClock.getTime())
            # the Routine "fixation" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'learning_loop'
        
        
        # --- Prepare to start Routine "post_section" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('post_section.started', globalClock.getTime())
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not ((bkNum < 5 ) and (bkNum % 5 != 0 ) )
        # Run 'Begin Routine' code from performance_post_section
        if len(bkRT) < 1:
            bkRT.append(0)
        if bkTrialNum < 1:
            bkTrialNum = 1
            
        bkRTmean = int(np.mean(bkRT)*1000)
        bkACC = bkCorrectNum / bkTrialNum * 100
        bkRTmeanstr = f"平均反應時間：{bkRTmean} ms"
        bkACCstr = f"正確率：{bkACC}%"
        sentence1 = "您在這一個段落的表現："
        sentence2 = "請按任意鍵繼續…"
        this_feedbacktext = f"""
        {sentence1}({bkTrialNum} trials, {bkNum} blocks)
        {bkRTmeanstr}
        {bkACCstr}
        {sentence2}
        """
        feedback_post_section.setText(this_feedbacktext)
        key_resp_4.keys = []
        key_resp_4.rt = []
        _key_resp_4_allKeys = []
        # keep track of which components have finished
        post_sectionComponents = [feedback_post_section, key_resp_4]
        for thisComponent in post_sectionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "post_section" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_post_section* updates
            
            # if feedback_post_section is starting this frame...
            if feedback_post_section.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_post_section.frameNStart = frameN  # exact frame index
                feedback_post_section.tStart = t  # local t and not account for scr refresh
                feedback_post_section.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_post_section, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_post_section.started')
                # update status
                feedback_post_section.status = STARTED
                feedback_post_section.setAutoDraw(True)
            
            # if feedback_post_section is active this frame...
            if feedback_post_section.status == STARTED:
                # update params
                pass
            
            # *key_resp_4* updates
            waitOnFlip = False
            
            # if key_resp_4 is starting this frame...
            if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_4.frameNStart = frameN  # exact frame index
                key_resp_4.tStart = t  # local t and not account for scr refresh
                key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_4.started')
                # update status
                key_resp_4.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_4.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_4.getKeys(keyList=['space', 'up','left', 'down', 'right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_4_allKeys.extend(theseKeys)
                if len(_key_resp_4_allKeys):
                    key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                    key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                    key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in post_sectionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "post_section" ---
        for thisComponent in post_sectionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('post_section.stopped', globalClock.getTime())
        # check responses
        if key_resp_4.keys in ['', [], None]:  # No response was made
            key_resp_4.keys = None
        learning_trials.addData('key_resp_4.keys',key_resp_4.keys)
        if key_resp_4.keys != None:  # we had a response
            learning_trials.addData('key_resp_4.rt', key_resp_4.rt)
            learning_trials.addData('key_resp_4.duration', key_resp_4.duration)
        # the Routine "post_section" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 1.0 repeats of 'learning_trials'
    
    
    # --- Prepare to start Routine "motor_testing_instruction" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('motor_testing_instruction.started', globalClock.getTime())
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # Run 'Begin Routine' code from pre_motor_testing
    bkNum = 0
    bkRT = []
    bkCorrectNum = 0
    bkTrialNum = 0
    
    # keep track of which components have finished
    motor_testing_instructionComponents = [spaceship3, motor_testing_instruction_text, key_resp_3]
    for thisComponent in motor_testing_instructionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "motor_testing_instruction" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *spaceship3* updates
        
        # if spaceship3 is starting this frame...
        if spaceship3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            spaceship3.frameNStart = frameN  # exact frame index
            spaceship3.tStart = t  # local t and not account for scr refresh
            spaceship3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(spaceship3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'spaceship3.started')
            # update status
            spaceship3.status = STARTED
            spaceship3.setAutoDraw(True)
        
        # if spaceship3 is active this frame...
        if spaceship3.status == STARTED:
            # update params
            pass
        
        # *motor_testing_instruction_text* updates
        
        # if motor_testing_instruction_text is starting this frame...
        if motor_testing_instruction_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            motor_testing_instruction_text.frameNStart = frameN  # exact frame index
            motor_testing_instruction_text.tStart = t  # local t and not account for scr refresh
            motor_testing_instruction_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(motor_testing_instruction_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'motor_testing_instruction_text.started')
            # update status
            motor_testing_instruction_text.status = STARTED
            motor_testing_instruction_text.setAutoDraw(True)
        
        # if motor_testing_instruction_text is active this frame...
        if motor_testing_instruction_text.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['space', 'up','left', 'down', 'right'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in motor_testing_instructionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "motor_testing_instruction" ---
    for thisComponent in motor_testing_instructionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('motor_testing_instruction.stopped', globalClock.getTime())
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "motor_testing_instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    motor_testing_trials = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('sequences/motor_testing_sequence_file_list.xlsx'),
        seed=None, name='motor_testing_trials')
    thisExp.addLoop(motor_testing_trials)  # add the loop to the experiment
    thisMotor_testing_trial = motor_testing_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisMotor_testing_trial.rgb)
    if thisMotor_testing_trial != None:
        for paramName in thisMotor_testing_trial:
            globals()[paramName] = thisMotor_testing_trial[paramName]
    
    for thisMotor_testing_trial in motor_testing_trials:
        currentLoop = motor_testing_trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisMotor_testing_trial.rgb)
        if thisMotor_testing_trial != None:
            for paramName in thisMotor_testing_trial:
                globals()[paramName] = thisMotor_testing_trial[paramName]
        
        # --- Prepare to start Routine "pre_section" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('pre_section.started', globalClock.getTime())
        # Run 'Begin Routine' code from performance_pre_section
        bkNum += 1
        # keep track of which components have finished
        pre_sectionComponents = []
        for thisComponent in pre_sectionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pre_section" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pre_sectionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pre_section" ---
        for thisComponent in pre_sectionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('pre_section.stopped', globalClock.getTime())
        # the Routine "pre_section" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        motor_testing_loop = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions(motor_testing_seq_files, selection='0:10'),
            seed=None, name='motor_testing_loop')
        thisExp.addLoop(motor_testing_loop)  # add the loop to the experiment
        thisMotor_testing_loop = motor_testing_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisMotor_testing_loop.rgb)
        if thisMotor_testing_loop != None:
            for paramName in thisMotor_testing_loop:
                globals()[paramName] = thisMotor_testing_loop[paramName]
        
        for thisMotor_testing_loop in motor_testing_loop:
            currentLoop = motor_testing_loop
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisMotor_testing_loop.rgb)
            if thisMotor_testing_loop != None:
                for paramName in thisMotor_testing_loop:
                    globals()[paramName] = thisMotor_testing_loop[paramName]
            
            # --- Prepare to start Routine "trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial.started', globalClock.getTime())
            arrowhead.setOri(orientation_degrees)
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            # keep track of which components have finished
            trialComponents = [arrowhead, key_resp]
            for thisComponent in trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *arrowhead* updates
                
                # if arrowhead is starting this frame...
                if arrowhead.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    arrowhead.frameNStart = frameN  # exact frame index
                    arrowhead.tStart = t  # local t and not account for scr refresh
                    arrowhead.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(arrowhead, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'arrowhead.started')
                    # update status
                    arrowhead.status = STARTED
                    arrowhead.setAutoDraw(True)
                
                # if arrowhead is active this frame...
                if arrowhead.status == STARTED:
                    # update params
                    pass
                
                # *key_resp* updates
                waitOnFlip = False
                
                # if key_resp is starting this frame...
                if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.started')
                    # update status
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['left','right','up','down'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                        key_resp.rt = _key_resp_allKeys[0].rt
                        key_resp.duration = _key_resp_allKeys[0].duration
                        # was this correct?
                        if (key_resp.keys == str(correct_answer_direction)) or (key_resp.keys == correct_answer_direction):
                            key_resp.corr = 1
                        else:
                            key_resp.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial.stopped', globalClock.getTime())
            # check responses
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
                # was no response the correct answer?!
                if str(correct_answer_direction).lower() == 'none':
                   key_resp.corr = 1;  # correct non-response
                else:
                   key_resp.corr = 0;  # failed to respond (incorrectly)
            # store data for motor_testing_loop (TrialHandler)
            motor_testing_loop.addData('key_resp.keys',key_resp.keys)
            motor_testing_loop.addData('key_resp.corr', key_resp.corr)
            if key_resp.keys != None:  # we had a response
                motor_testing_loop.addData('key_resp.rt', key_resp.rt)
                motor_testing_loop.addData('key_resp.duration', key_resp.duration)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "fixation" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('fixation.started', globalClock.getTime())
            # Run 'Begin Routine' code from code
            if key_resp.keys == correct_answer_direction:
                feedbackstr ="正確"
                bkCorrectNum += 1
            else:
                feedbackstr = "錯誤"
                
            bkRT.append(key_resp.rt)
            bkTrialNum+=1
            # keep track of which components have finished
            fixationComponents = [cross_2]
            for thisComponent in fixationComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *cross_2* updates
                
                # if cross_2 is starting this frame...
                if cross_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cross_2.frameNStart = frameN  # exact frame index
                    cross_2.tStart = t  # local t and not account for scr refresh
                    cross_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cross_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_2.started')
                    # update status
                    cross_2.status = STARTED
                    cross_2.setAutoDraw(True)
                
                # if cross_2 is active this frame...
                if cross_2.status == STARTED:
                    # update params
                    pass
                
                # if cross_2 is stopping this frame...
                if cross_2.status == STARTED:
                    if frameN >= 8:
                        # keep track of stop time/frame for later
                        cross_2.tStop = t  # not accounting for scr refresh
                        cross_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cross_2.stopped')
                        # update status
                        cross_2.status = FINISHED
                        cross_2.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixationComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation" ---
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('fixation.stopped', globalClock.getTime())
            # the Routine "fixation" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'motor_testing_loop'
        
        
        # --- Prepare to start Routine "post_section" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('post_section.started', globalClock.getTime())
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not ((bkNum < 5 ) and (bkNum % 5 != 0 ) )
        # Run 'Begin Routine' code from performance_post_section
        if len(bkRT) < 1:
            bkRT.append(0)
        if bkTrialNum < 1:
            bkTrialNum = 1
            
        bkRTmean = int(np.mean(bkRT)*1000)
        bkACC = bkCorrectNum / bkTrialNum * 100
        bkRTmeanstr = f"平均反應時間：{bkRTmean} ms"
        bkACCstr = f"正確率：{bkACC}%"
        sentence1 = "您在這一個段落的表現："
        sentence2 = "請按任意鍵繼續…"
        this_feedbacktext = f"""
        {sentence1}({bkTrialNum} trials, {bkNum} blocks)
        {bkRTmeanstr}
        {bkACCstr}
        {sentence2}
        """
        feedback_post_section.setText(this_feedbacktext)
        key_resp_4.keys = []
        key_resp_4.rt = []
        _key_resp_4_allKeys = []
        # keep track of which components have finished
        post_sectionComponents = [feedback_post_section, key_resp_4]
        for thisComponent in post_sectionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "post_section" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_post_section* updates
            
            # if feedback_post_section is starting this frame...
            if feedback_post_section.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_post_section.frameNStart = frameN  # exact frame index
                feedback_post_section.tStart = t  # local t and not account for scr refresh
                feedback_post_section.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_post_section, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_post_section.started')
                # update status
                feedback_post_section.status = STARTED
                feedback_post_section.setAutoDraw(True)
            
            # if feedback_post_section is active this frame...
            if feedback_post_section.status == STARTED:
                # update params
                pass
            
            # *key_resp_4* updates
            waitOnFlip = False
            
            # if key_resp_4 is starting this frame...
            if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_4.frameNStart = frameN  # exact frame index
                key_resp_4.tStart = t  # local t and not account for scr refresh
                key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_4.started')
                # update status
                key_resp_4.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_4.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_4.getKeys(keyList=['space', 'up','left', 'down', 'right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_4_allKeys.extend(theseKeys)
                if len(_key_resp_4_allKeys):
                    key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                    key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                    key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in post_sectionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "post_section" ---
        for thisComponent in post_sectionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('post_section.stopped', globalClock.getTime())
        # check responses
        if key_resp_4.keys in ['', [], None]:  # No response was made
            key_resp_4.keys = None
        motor_testing_trials.addData('key_resp_4.keys',key_resp_4.keys)
        if key_resp_4.keys != None:  # we had a response
            motor_testing_trials.addData('key_resp_4.rt', key_resp_4.rt)
            motor_testing_trials.addData('key_resp_4.duration', key_resp_4.duration)
        # the Routine "post_section" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'motor_testing_trials'
    
    
    # --- Prepare to start Routine "percept_testing_instruction" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('percept_testing_instruction.started', globalClock.getTime())
    key_resp_5.keys = []
    key_resp_5.rt = []
    _key_resp_5_allKeys = []
    # Run 'Begin Routine' code from pre_percept
    bkNum = 0
    bkRT = []
    bkCorrectNum = 0
    bkTrialNum = 0
    
    # keep track of which components have finished
    percept_testing_instructionComponents = [spaceship4, key_resp_5, percept_testing_instruction_text]
    for thisComponent in percept_testing_instructionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "percept_testing_instruction" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *spaceship4* updates
        
        # if spaceship4 is starting this frame...
        if spaceship4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            spaceship4.frameNStart = frameN  # exact frame index
            spaceship4.tStart = t  # local t and not account for scr refresh
            spaceship4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(spaceship4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'spaceship4.started')
            # update status
            spaceship4.status = STARTED
            spaceship4.setAutoDraw(True)
        
        # if spaceship4 is active this frame...
        if spaceship4.status == STARTED:
            # update params
            pass
        
        # *key_resp_5* updates
        waitOnFlip = False
        
        # if key_resp_5 is starting this frame...
        if key_resp_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_5.frameNStart = frameN  # exact frame index
            key_resp_5.tStart = t  # local t and not account for scr refresh
            key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_5.started')
            # update status
            key_resp_5.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_5.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_5.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_5_allKeys.extend(theseKeys)
            if len(_key_resp_5_allKeys):
                key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                key_resp_5.rt = _key_resp_5_allKeys[-1].rt
                key_resp_5.duration = _key_resp_5_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *percept_testing_instruction_text* updates
        
        # if percept_testing_instruction_text is starting this frame...
        if percept_testing_instruction_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            percept_testing_instruction_text.frameNStart = frameN  # exact frame index
            percept_testing_instruction_text.tStart = t  # local t and not account for scr refresh
            percept_testing_instruction_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(percept_testing_instruction_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'percept_testing_instruction_text.started')
            # update status
            percept_testing_instruction_text.status = STARTED
            percept_testing_instruction_text.setAutoDraw(True)
        
        # if percept_testing_instruction_text is active this frame...
        if percept_testing_instruction_text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in percept_testing_instructionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "percept_testing_instruction" ---
    for thisComponent in percept_testing_instructionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('percept_testing_instruction.stopped', globalClock.getTime())
    # check responses
    if key_resp_5.keys in ['', [], None]:  # No response was made
        key_resp_5.keys = None
    thisExp.addData('key_resp_5.keys',key_resp_5.keys)
    if key_resp_5.keys != None:  # we had a response
        thisExp.addData('key_resp_5.rt', key_resp_5.rt)
        thisExp.addData('key_resp_5.duration', key_resp_5.duration)
    thisExp.nextEntry()
    # the Routine "percept_testing_instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    percept_testing_trials = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('sequences/percept_testing_sequence_file_list.xlsx'),
        seed=None, name='percept_testing_trials')
    thisExp.addLoop(percept_testing_trials)  # add the loop to the experiment
    thisPercept_testing_trial = percept_testing_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPercept_testing_trial.rgb)
    if thisPercept_testing_trial != None:
        for paramName in thisPercept_testing_trial:
            globals()[paramName] = thisPercept_testing_trial[paramName]
    
    for thisPercept_testing_trial in percept_testing_trials:
        currentLoop = percept_testing_trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisPercept_testing_trial.rgb)
        if thisPercept_testing_trial != None:
            for paramName in thisPercept_testing_trial:
                globals()[paramName] = thisPercept_testing_trial[paramName]
        
        # --- Prepare to start Routine "pre_section" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('pre_section.started', globalClock.getTime())
        # Run 'Begin Routine' code from performance_pre_section
        bkNum += 1
        # keep track of which components have finished
        pre_sectionComponents = []
        for thisComponent in pre_sectionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pre_section" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pre_sectionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pre_section" ---
        for thisComponent in pre_sectionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('pre_section.stopped', globalClock.getTime())
        # the Routine "pre_section" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        percept_testing_loop = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions(percept_testing_seq_files, selection='0:10'),
            seed=None, name='percept_testing_loop')
        thisExp.addLoop(percept_testing_loop)  # add the loop to the experiment
        thisPercept_testing_loop = percept_testing_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPercept_testing_loop.rgb)
        if thisPercept_testing_loop != None:
            for paramName in thisPercept_testing_loop:
                globals()[paramName] = thisPercept_testing_loop[paramName]
        
        for thisPercept_testing_loop in percept_testing_loop:
            currentLoop = percept_testing_loop
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisPercept_testing_loop.rgb)
            if thisPercept_testing_loop != None:
                for paramName in thisPercept_testing_loop:
                    globals()[paramName] = thisPercept_testing_loop[paramName]
            
            # --- Prepare to start Routine "trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial.started', globalClock.getTime())
            arrowhead.setOri(orientation_degrees)
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            # keep track of which components have finished
            trialComponents = [arrowhead, key_resp]
            for thisComponent in trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *arrowhead* updates
                
                # if arrowhead is starting this frame...
                if arrowhead.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    arrowhead.frameNStart = frameN  # exact frame index
                    arrowhead.tStart = t  # local t and not account for scr refresh
                    arrowhead.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(arrowhead, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'arrowhead.started')
                    # update status
                    arrowhead.status = STARTED
                    arrowhead.setAutoDraw(True)
                
                # if arrowhead is active this frame...
                if arrowhead.status == STARTED:
                    # update params
                    pass
                
                # *key_resp* updates
                waitOnFlip = False
                
                # if key_resp is starting this frame...
                if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.started')
                    # update status
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['left','right','up','down'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                        key_resp.rt = _key_resp_allKeys[0].rt
                        key_resp.duration = _key_resp_allKeys[0].duration
                        # was this correct?
                        if (key_resp.keys == str(correct_answer_direction)) or (key_resp.keys == correct_answer_direction):
                            key_resp.corr = 1
                        else:
                            key_resp.corr = 0
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial.stopped', globalClock.getTime())
            # check responses
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
                # was no response the correct answer?!
                if str(correct_answer_direction).lower() == 'none':
                   key_resp.corr = 1;  # correct non-response
                else:
                   key_resp.corr = 0;  # failed to respond (incorrectly)
            # store data for percept_testing_loop (TrialHandler)
            percept_testing_loop.addData('key_resp.keys',key_resp.keys)
            percept_testing_loop.addData('key_resp.corr', key_resp.corr)
            if key_resp.keys != None:  # we had a response
                percept_testing_loop.addData('key_resp.rt', key_resp.rt)
                percept_testing_loop.addData('key_resp.duration', key_resp.duration)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "fixation" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('fixation.started', globalClock.getTime())
            # Run 'Begin Routine' code from code
            if key_resp.keys == correct_answer_direction:
                feedbackstr ="正確"
                bkCorrectNum += 1
            else:
                feedbackstr = "錯誤"
                
            bkRT.append(key_resp.rt)
            bkTrialNum+=1
            # keep track of which components have finished
            fixationComponents = [cross_2]
            for thisComponent in fixationComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *cross_2* updates
                
                # if cross_2 is starting this frame...
                if cross_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cross_2.frameNStart = frameN  # exact frame index
                    cross_2.tStart = t  # local t and not account for scr refresh
                    cross_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cross_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_2.started')
                    # update status
                    cross_2.status = STARTED
                    cross_2.setAutoDraw(True)
                
                # if cross_2 is active this frame...
                if cross_2.status == STARTED:
                    # update params
                    pass
                
                # if cross_2 is stopping this frame...
                if cross_2.status == STARTED:
                    if frameN >= 8:
                        # keep track of stop time/frame for later
                        cross_2.tStop = t  # not accounting for scr refresh
                        cross_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cross_2.stopped')
                        # update status
                        cross_2.status = FINISHED
                        cross_2.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixationComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation" ---
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('fixation.stopped', globalClock.getTime())
            # the Routine "fixation" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'percept_testing_loop'
        
        
        # --- Prepare to start Routine "post_section" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('post_section.started', globalClock.getTime())
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not ((bkNum < 5 ) and (bkNum % 5 != 0 ) )
        # Run 'Begin Routine' code from performance_post_section
        if len(bkRT) < 1:
            bkRT.append(0)
        if bkTrialNum < 1:
            bkTrialNum = 1
            
        bkRTmean = int(np.mean(bkRT)*1000)
        bkACC = bkCorrectNum / bkTrialNum * 100
        bkRTmeanstr = f"平均反應時間：{bkRTmean} ms"
        bkACCstr = f"正確率：{bkACC}%"
        sentence1 = "您在這一個段落的表現："
        sentence2 = "請按任意鍵繼續…"
        this_feedbacktext = f"""
        {sentence1}({bkTrialNum} trials, {bkNum} blocks)
        {bkRTmeanstr}
        {bkACCstr}
        {sentence2}
        """
        feedback_post_section.setText(this_feedbacktext)
        key_resp_4.keys = []
        key_resp_4.rt = []
        _key_resp_4_allKeys = []
        # keep track of which components have finished
        post_sectionComponents = [feedback_post_section, key_resp_4]
        for thisComponent in post_sectionComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "post_section" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_post_section* updates
            
            # if feedback_post_section is starting this frame...
            if feedback_post_section.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_post_section.frameNStart = frameN  # exact frame index
                feedback_post_section.tStart = t  # local t and not account for scr refresh
                feedback_post_section.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_post_section, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_post_section.started')
                # update status
                feedback_post_section.status = STARTED
                feedback_post_section.setAutoDraw(True)
            
            # if feedback_post_section is active this frame...
            if feedback_post_section.status == STARTED:
                # update params
                pass
            
            # *key_resp_4* updates
            waitOnFlip = False
            
            # if key_resp_4 is starting this frame...
            if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_4.frameNStart = frameN  # exact frame index
                key_resp_4.tStart = t  # local t and not account for scr refresh
                key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_4.started')
                # update status
                key_resp_4.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_4.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_4.getKeys(keyList=['space', 'up','left', 'down', 'right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_4_allKeys.extend(theseKeys)
                if len(_key_resp_4_allKeys):
                    key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                    key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                    key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in post_sectionComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "post_section" ---
        for thisComponent in post_sectionComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('post_section.stopped', globalClock.getTime())
        # check responses
        if key_resp_4.keys in ['', [], None]:  # No response was made
            key_resp_4.keys = None
        percept_testing_trials.addData('key_resp_4.keys',key_resp_4.keys)
        if key_resp_4.keys != None:  # we had a response
            percept_testing_trials.addData('key_resp_4.rt', key_resp_4.rt)
            percept_testing_trials.addData('key_resp_4.duration', key_resp_4.duration)
        # the Routine "post_section" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'percept_testing_trials'
    
    
    # --- Prepare to start Routine "thankyouscreen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('thankyouscreen.started', globalClock.getTime())
    key_resp_6.keys = []
    key_resp_6.rt = []
    _key_resp_6_allKeys = []
    # keep track of which components have finished
    thankyouscreenComponents = [text, key_resp_6]
    for thisComponent in thankyouscreenComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thankyouscreen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp_6* updates
        waitOnFlip = False
        
        # if key_resp_6 is starting this frame...
        if key_resp_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_6.frameNStart = frameN  # exact frame index
            key_resp_6.tStart = t  # local t and not account for scr refresh
            key_resp_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_6.started')
            # update status
            key_resp_6.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_6.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_6.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_6.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_6_allKeys.extend(theseKeys)
            if len(_key_resp_6_allKeys):
                key_resp_6.keys = _key_resp_6_allKeys[-1].name  # just the last key pressed
                key_resp_6.rt = _key_resp_6_allKeys[-1].rt
                key_resp_6.duration = _key_resp_6_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thankyouscreenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thankyouscreen" ---
    for thisComponent in thankyouscreenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('thankyouscreen.stopped', globalClock.getTime())
    # check responses
    if key_resp_6.keys in ['', [], None]:  # No response was made
        key_resp_6.keys = None
    thisExp.addData('key_resp_6.keys',key_resp_6.keys)
    if key_resp_6.keys != None:  # we had a response
        thisExp.addData('key_resp_6.rt', key_resp_6.rt)
        thisExp.addData('key_resp_6.duration', key_resp_6.duration)
    thisExp.nextEntry()
    # the Routine "thankyouscreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
