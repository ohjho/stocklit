################################################################################
# ref: https://discuss.streamlit.io/t/multiple-apps-from-a-folder-file-list/181/4
################################################################################

import streamlit as st
import os, sys, importlib.util

# Parse command-line arguments.
if len(sys.argv) > 1:
    folder = os.path.abspath(sys.argv[1])
else:
    folder = os.path.abspath(os.getcwd())

# Get filenames for all files in this path, excluding this script.

this_file = os.path.abspath(__file__)
fnames = []

for basename in os.listdir(folder):
    fname = os.path.join(folder, basename)

    if fname.endswith('.py') and fname != this_file:
        fnames.append(fname)

# Make a UI to run different files.

fname_to_run = st.sidebar.selectbox('Select an app', fnames)

# Create module from filepath and put in sys.modules, so Streamlit knows
# to watch it for changes.

fake_module_count = 0

def load_module(filepath):
    global fake_module_count

    modulename = '_dont_care_%s' % fake_module_count
    spec = importlib.util.spec_from_file_location(modulename, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modulename] = module

    fake_module_count += 1

# Run the selected file.

with open(fname_to_run) as f:
    load_module(fname_to_run)
    filebody = f.read()

exec(filebody, {})
