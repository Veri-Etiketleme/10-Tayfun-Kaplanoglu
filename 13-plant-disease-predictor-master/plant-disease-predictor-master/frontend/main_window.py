import PySimpleGUI as gui
from frontend.image_viewer import convert_to_bytes, IMG_SIZE
from backend.classifier import classify

WINDOW_WIDTH, WINDOW_HEIGHT = 655, 365

PROGRAM_TITLE = 'Plant Disease Predictor'
MODELS_FOLDER = 'backend/models'
TEST_IMAGE_FOLDER = 'testdata'
THEME = 'Material 2'

image_file_types = [
    ("JPEG", ("*.JPG", "*.jpg")),
    ("PNG", ("*.PNG", "*.png")),
    ("All files", "*.*")
]

gui.theme(THEME)

program_title = [
    [gui.Text(PROGRAM_TITLE, size=(WINDOW_WIDTH, 1), font=('Comic Sans', 20), justification='center')],
    [gui.Text('_'*42, size=(WINDOW_WIDTH, 2), justification='center')] 
]

model_image_frame = [
    [gui.Text('Model:'), gui.Combo([], gui.user_settings_get_entry('-model-', ''), key='-model-', size=(30, 1),
                                   auto_size_text=False), gui.FileBrowse(initial_folder=MODELS_FOLDER)],
    [gui.Text('Image:'), gui.Combo([], gui.user_settings_get_entry('-image-', ''), key='-image-', size=(30, 1),
                                   auto_size_text=False), gui.FileBrowse(initial_folder=TEST_IMAGE_FOLDER, file_types=image_file_types)]
]

user_inputs = [
    [gui.Frame('Select Model and Input image', model_image_frame, font='Any 12', title_color='blue')],
    [gui.Frame('',[[gui.Multiline(background_color='White',text_color='Black' , size = (30,3), font='Any 14', key='-result-', no_scrollbar=True, write_only=True)]])]
]

image_viwer = [
    [gui.Image(size=IMG_SIZE, key='-imageViewer-')]
]

main_window_layout = [
    [gui.Frame('',layout=program_title)],
    [gui.Column(layout=user_inputs), gui.VerticalSeparator(), gui.Column(layout=image_viwer)],
    [gui.Button('Ok', key='-ok-'), gui.Button('Exit', key='-exit-')]
]

main_window = gui.Window(
    title=PROGRAM_TITLE,
    layout=main_window_layout,
    size=(WINDOW_WIDTH, WINDOW_HEIGHT),
    auto_size_text=True,
    resizable=False,
    enable_close_attempted_event=True,
    return_keyboard_events=True
)

def run():
    while True:
        event, values = main_window.read()

        if (event == gui.WINDOW_CLOSE_ATTEMPTED_EVENT or event == '-exit-') and gui.popup_yes_no('Do you really want to exit?') == 'Yes':
            gui.user_settings_set_entry('-model-', values['-model-'])
            gui.user_settings_set_entry('-image-', values['-image-'])
            break    
        else:
            model = values['-model-']
            image = values['-image-']
            if  model and image :
                try:
                    image_bytes, image_data = convert_to_bytes(image)
                    main_window['-imageViewer-'].update(data=image_bytes)
                    result = classify(image_data, model)
                    main_window['-result-'].update(result)
                except Exception as E:
                    gui.popup("Exception : {0} : {1}".format(type(E).__name__,E))
            else:
                gui.popup('Please provide valid Model name and Image')
    main_window.close()
