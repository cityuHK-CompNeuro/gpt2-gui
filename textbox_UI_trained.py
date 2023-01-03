import os
import sys
import torch
import random
import argparse
import time

import numpy as np
import PySimpleGUI as sg

from transformers import GPT2LMHeadModel,  GPT2Tokenizer

def format_input_information(values, state_dict):
    information = "You Entered:\n"
    information += values['-NAME-'] + '\n\n'
    information += "Generated Text:\n"

    try:
        # information += get_text_suggestion(model, values['-NAME-'])
        # print(values['-NAME-'])
        # print(text_generator(state_dict, values['-NAME-']))
        text_list_out = text_generator(state_dict, values['-NAME-'], length=250)
        information += text_list_out[0]
    except NameError:
        print(f'Logging Error Info: {NameError}')
    
    return information

def get_text_suggestion(model, input_text):
    text_out = model(input_text)
    return text_out

def store_previous_prompts(values, prompts):
    prompts.append(format_input_information(values))

def reservations_window(reservations_array):
    # Layout is here because it must be "new" every time you open the window.
    reservations_window_layout = [[sg.Listbox(values=reservations_array, size=(200, 5), select_mode='single', key='-DESTINATION-')],
                                  [sg.Button("Exit")]
                                 ]
    reservations_window = sg.Window("Reservations Window", reservations_window_layout, modal=True)
    while True:
        event, values = reservations_window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    reservations_window.close() 

def text_generator(model, text_in, **kwargs):
    length = kwargs.get('length', -1)

    generated = torch.tensor(tokenizer.encode(text_in)).unsqueeze(0)
    generated = generated.to(device)

    sample_outputs = model.generate(generated, 
                                #bos_token_id=random.randint(1,30000),
                                do_sample=True,   
                                top_k=10, 
                                max_length = length,
                                top_p=0.97, 
                                num_return_sequences=1)
    
    text_list_out = [tokenizer.decode(s_o, skip_special_tokens=True) for s_o in sample_outputs]

    return text_list_out

## Main
sg.theme('DarkGray')

layout = [[sg.Text("Enter Text Prompt:"), sg.Input(key='-NAME-', do_not_clear=True, size=(50, 1))],
          [sg.Text('Type your query in the box. Suggested Text Will Appear Here', key='-TEXT-')],
          [sg.Button('Generate Text'), sg.Exit()]]

# Store Previous Prompts and Outputs
prompts = []

# Create the window
window = sg.Window("Text Input", layout)

model_path = 'D:\\Coding\\sandbox\\gpt2_train\\model_save\\20221228_2153'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if os.path.exists(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model.to(device)

else:
    print('Please download gpt2-pytorch_model.bin')
    sys.exit()

old_text = ''

# UI Event loop
while True:
    event, values = window.read(timeout=1)
    # End program if user closes window
    if event in (None, sg.WIN_CLOSED, 'Exit'):
        break
    elif event == 'Generate Text':
        # print(state_dict)
        # store_previous_prompts(values, prompts)
        sg.popup(format_input_information(values, model))
    elif event == sg.TIMEOUT_KEY:
        if old_text != values['-NAME-']: # and  ' ' in values['-NAME-']:
            suggested_text_list = text_generator(model, values['-NAME-'], length=10)
            # oneword_updater = suggested_text.split()[:2]
            # oneword_out = ' '.join(map(str,oneword_updater))
            window['-TEXT-'].update(f"{suggested_text_list[0]}")
            old_text = values['-NAME-']
            # time.sleep(0.3)

window.close()