import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from utilities.data_operations import calculate_statistics, generate_wave, predict_timeseries
from utilities.static_text import print_static_text
from utilities.visual_operations import plot_dataset, populate_sidebar_get_setup

# Setting up global parameters - the challenge is to predict the next 8 points
TEST_LEN = 8
MY_PALLETE = ['#1e4a76', '#7dc0f7', '#ff7c0a']

# Set up menu and print static text
st.set_page_config(page_title='ARIMA Playground')
pages = {'Intro': 0,
         'What is the ARIMA model?': 1,
         'The playground': 2}
choice = pages[st.sidebar.radio("Select chapter:", tuple(pages.keys()), 2)]
print_static_text(choice)

# Show the playground if it's selected
if choice == 2:
    data_setup, model_setup = populate_sidebar_get_setup()
    points_granularity = data_setup['points_granularity']
    # Fetch user input and generate dataset
    st.subheader("Generated data")
    X = np.array(range(data_setup['length'] + TEST_LEN)) * points_granularity
    Y = generate_wave(X, data_setup)

    # Train, test split
    X_test, Y_test = X[data_setup['length'] - 1:], Y[data_setup['length'] - 1:]
    X, Y = X[:data_setup['length']], Y[:data_setup['length']]

    # Plot main graph
    st.write("The plot presents generated data. Use the sliders on the left to modify the curve. "
             "You can turn off plotting the dataset to improve difficulty.")
    test_tuple = (X_test, Y_test) if data_setup['show_test'] else None
    plot_dataset((X, Y), test_tuple, pallete=MY_PALLETE)

    # Show some insights into the generated dataset
    if data_setup['validate']:
        st.subheader("Dataset validation")
        st.text(f'Train data average value: {np.mean(X)}')
        st.write("It's essential to understand what happens in the data to configure the model hyperparameters. "
                 "ACF and PACF graphs are helpful to see how dynamic the dataset is.")
        fig, ax = plt.subplots()
        acf_original = plot_acf(X, lags=int(len(X) / 2) - 1)
        st.pyplot(acf_original)
        pacf_original = plot_pacf(X, ax, lags=int(len(X) / 2) - 1)
        st.pyplot(pacf_original)

    # Predict the future using selected options, plot the results, show predictions error
    if model_setup['train']:
        st.subheader("Predictions")
        model_setup['values'] = list(Y)
        predictions = predict_timeseries(Y, steps=TEST_LEN, **model_setup)

        # To make graph continuous, include the last training point in the graph.
        # X values for the test and pred dataset is the same
        X_pred = X_test
        Y_pred = np.append(Y[-1], predictions)

        # Plot results
        plot_dataset((X, Y), (X_test, Y_test), (X_pred, Y_pred), pallete=MY_PALLETE)
        stats = calculate_statistics(Y_test[1:], Y_pred[1:])
        st.write("Is the prediction too good? Try to make the dataset more challenging to predict!")
        st.write("Decide which loss function best meets your needs, as the generator can produce data "
                 "with different characteristics.")
        st.write(stats)
