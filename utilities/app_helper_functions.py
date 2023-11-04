import numpy as np
import streamlit as st


def populate_sidebar_get_setup():
    # Make buttons pretty
    local_css("utilities/button_style.css")

    # Collect dataset set-up
    data = dict()
    st.sidebar.header('Dataset:')
    data['length'] = st.sidebar.slider('Training data points', 10, 50, 20)
    data['points_granularity'] = st.sidebar.slider('Distance between points', 0.1, 2.0, 0.6)
    st.sidebar.subheader('Wave parameters:')
    data['amplitude'] = st.sidebar.slider('Wave amplitude (you can set to 0 to disable wave)', 0.00, 1.75, 1.0)
    data['period'] = st.sidebar.slider('Period of the wave', 0.50, 2.0, 1.0)
    data['amplitude_growth'] = st.sidebar.slider('Wave amplitude growth', -0.2, 0.3, 0.1)
    st.sidebar.subheader('Other dataset options:')
    data['growth'] = st.sidebar.slider('Values growth in function of time', -0.5, 0.5, 0.0)
    data['noise'] = st.sidebar.slider('Random noise', 0.0, 1.0, 0.2)
    data['make_log'] = st.sidebar.radio('Use logarithm', ['No', 'Yes']) == 'Yes'
    # if we want to use log, let's make sure we always have positive values
    if data['make_log']:
        disabled, index = True, 1
    else:
        disabled, index = False, 0
    data['make_values_positive'] = st.sidebar.radio('Make values positive', ['No', 'Yes'], disabled=disabled,
                                                    index=index) == 'Yes'
    data['show_test'] = st.sidebar.radio('Secret: show test dataset', ['Yes', 'No']) == 'Yes'
    data['validate'] = st.sidebar.button('Validate dataset')

    # Collect model set-up
    model = dict()
    st.sidebar.header('Model setup')
    model['ar'] = st.sidebar.slider('AR - Lag order', 0, 10, 2)
    model['i'] = st.sidebar.slider('I - Degree of differencing', 0, 10, 0)
    model['ma'] = st.sidebar.slider('MA - Size of the moving average window', 0, 10, 1)
    model['train'] = st.sidebar.button('Train and Predict')

    return data, model


def generate_wave(x_set, params):
    # Generate wave based on sin transformations
    wave = np.sin(x_set / params['period']) * (1 + params['amplitude_growth'] * x_set) * params[
        'amplitude'] + x_set * params['growth'] + params['noise'] * np.random.randn(len(x_set))

    if params['make_values_positive'] and min(wave) < 1:
        wave = wave - min(wave) + 1

    if params['make_log']:
        wave = np.log(wave)

    return wave


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
