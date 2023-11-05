import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from utilities.app_helper_functions import generate_wave, plot_dataset, populate_sidebar_get_setup
from utilities.predict import calculate_statistics, predict_timeseries

# Setting up global parameters - the challenge is to predict the next 8 points
TEST_LEN = 8
MY_PALLETE = ['#1e4a76', '#7dc0f7', '#ff7c0a']

# Create chapter menu
# TODO: Finish the second page - ARIMA model theory
st.set_page_config(page_title='ARIMA Playground')
pages = {'Intro': 0,
         # TODO: 'ARIMA model': 1
         'The playground': 2}
choice = pages[st.sidebar.radio("Select chapter:", tuple(pages.keys()), 1)]

if choice == 0:
    st.title("ARIMA playground")
    st.subheader("Hi!")
    st.image('Krzysztof.jpg', width=600, caption="It's me and Concorde")
    st.write("""\n
            Krzysztof Chodara here! \n
            I recently noticed that despite having a lot of experience in data science and time-series forecasting,
             I haven't come across a practical use of ARIMA model. Typically I have used CNN or regression
             for particular points in time. So I created this webpage to play around with ARIMA model.
            """)

    st.subheader("The purpose")
    st.write("""\n
        The goal of this app is to allow an user to experiment easily with ARIMA models. Thanks to that, the app helps
         to understand: \n
        - When is it efficient to use the model
        - Which scenarios are more straightforward for the model to predict and which are more difficult
        - How the noise interrupts the predictions
        - How important is the right balance between autoregression, differencing and moving average parts of the model
        - How we can set up the parts from the previous point
        - Understanding that the increasing hyperparameters' value in the model configuration doesn't always lead to
          better performance
        """)

    st.subheader("Typical use case")
    st.write("""
        1. Create a synthetic dataset using wide range of choices to define the charactericts
        2. See and validate the dataset to understand the data 
        3. Set up model hyperparamets
        4. Train the model and make predictions
        5. Compare the predicted values with the actual values
        \n""")

    st.subheader(""" \n Use the radio buttons on the left to navigate between chapters \n \n \n \n""")

elif choice == 1:
    # To be finished up later
    st.title(""" ARIMA model""")
    st.subheader("What is a timeseries modeling?")
    st.subheader("What it differs from other time-series forecasting models?")
    st.subheader("How is it build?")
    st.subheader("How to use it efficiently?")
    st.write("""\n
                 123
                 456 \n
                 789 """)

else:
    # Print instructions
    data_setup, model_setup = populate_sidebar_get_setup()
    points_granularity = data_setup['points_granularity']
    st.text("""Note: Work in progress - new updates soon""")
    st.subheader("Instructions:")
    st.write("""
            1. Modify the dataset by using sliders in the dataset group in the sidebar.
            2. Use the "Validate dataset" button to check if the dataset is stationary. Scroll sidebar down to 
               see the buttons.
            3. Based on the findings set up the model parameters and press the "Train and predict" button. 
               You will see the predictions below the original graph.
            4. If the model predictions are off - modify the model. If the predictions are accurate - 
               make the dataset more challening. 
            5. Good luck and have fun!
            \n""")

    # Fetch user input and generate dataset
    st.subheader("Generated data:")
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
        st.subheader("Dataset validation:")
        st.text(f'Train data average value: {np.mean(X)}')
        st.write("To set-up the model hyperparameters it's important to understand what happens in the data. "
                 "ACF and PACF graphs are helpful to see how dynamic is the dataset")
        fig, ax = plt.subplots()
        acf_original = plot_acf(X, lags=int(len(X) / 2) - 1)
        st.pyplot(acf_original)
        pacf_original = plot_pacf(X, ax, lags=int(len(X) / 2) - 1)
        st.pyplot(pacf_original)

    # Predict the future using selected options, plot the results, show predictions error
    if model_setup['train']:
        st.subheader("Predictions:")
        model_setup['values'] = list(Y)
        predictions = predict_timeseries(Y, steps=TEST_LEN, **model_setup)

        # To make graph continuous, include the last training point in the graph.
        # X values for the test and pred dataset is the same
        X_pred = X_test
        Y_pred = np.append(Y[-1], predictions)

        # Plot results
        plot_dataset((X, Y), (X_test, Y_test), (X_pred, Y_pred), pallete=MY_PALLETE)
        stats = calculate_statistics(Y_test[1:], Y_pred[1:])
        st.write("The prediction is too good? Try to make the dataset more challenging to predict.")
        st.write("Since the generator can produce data with various characteristics decide which loss function"
                 " matches your needs at best.")
        st.write(stats)
