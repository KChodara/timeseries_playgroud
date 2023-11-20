import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
    data['amplitude'] = st.sidebar.slider('Wave amplitude (Set to 0 to disable wave)', 0.00, 1.75, 1.0)
    data['period'] = st.sidebar.slider('Period of the wave', 0.50, 2.0, 1.0)
    data['amplitude_growth'] = st.sidebar.slider('Wave amplitude growth', -0.2, 0.3, 0.1)
    st.sidebar.subheader('Other dataset options:')
    data['noise'] = st.sidebar.slider('Random noise', 0.0, 1.0, 0.1)
    data['growth'] = st.sidebar.slider('Values growth in function of time', -0.5, 0.5, 0.0)
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


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def plot_dataset(data_train, data_test=None, data_pred=None, pallete=('black', 'blue', 'red')):
    legend_handles = []
    fig, ax = plt.subplots()
    line_train = sns.lineplot(x=data_train[0], y=data_train[1], color=pallete[0], label='Train')
    line_train.set_label('Train')
    if data_test:
        line_test = sns.lineplot(x=data_test[0], y=data_test[1], color=pallete[1], linestyle=':', label='Test')
        legend_handles.append(line_test)
    # To keep graph tidy - show points on the original data only if there is no prediction data.
    # But always show the points on the prediction data
    if data_pred is None:
        sns.scatterplot(x=data_train[0], y=data_train[1], color=pallete[0], size=0.1, marker='x', legend=False)
        if data_test:
            sns.scatterplot(x=data_test[0], y=data_test[1], color='k', size=1.1, marker='x', legend=False)
    else:
        line_pred = sns.lineplot(x=data_pred[0], y=data_pred[1], color=pallete[2], label='Predictions')
        legend_handles.append(line_pred)
        sns.scatterplot(x=data_pred[0][1:], y=data_pred[1][1:], color=pallete[2], size=0.2, legend=False)

    # To help user understand the amplitude scale during data generation - make the y-axis limits a little bit leaky.
    # But when showing the predictions use the default scale
    if data_pred is None:
        if data_test:
            y_lim_data = np.concatenate((data_train[1], data_test[1]))
        else:
            y_lim_data = data_train[1]
        plt.ylim(min(0, min(y_lim_data) * 1.1, max(1, max(y_lim_data) * 1.1)))

    plt.legend()
    plt.xlabel('Time [minutes]')
    plt.ylabel('Sample value')
    st.pyplot(fig)
