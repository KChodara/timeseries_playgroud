import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from utilities.visual_operations import plot_dataset, populate_sidebar_get_setup
from utilities.data_operations import calculate_statistics, generate_wave, predict_timeseries

# Setting up global parameters - the challenge is to predict the next 8 points
TEST_LEN = 8
MY_PALLETE = ['#1e4a76', '#7dc0f7', '#ff7c0a']

# Create chapter menu
st.set_page_config(page_title='ARIMA Playground')
pages = {'Page intro': 0,
         'What is ARIMA model?': 1,
         'The playground': 2}
choice = pages[st.sidebar.radio("Select chapter:", tuple(pages.keys()), 1)]

if choice == 0:
    st.title("ARIMA playground")
    st.image('Krzysztof.jpg', width=400, caption="It's me and Concorde")
    st.write("""\n
            Hi! Krzysztof Chodara here! \n
            I am a Senior Data Scientist with an extensive background in data science and time series forecasting.
             However, until now, I have yet to encounter a practical use of the ARIMA model. I have typically 
             used CNNs or regressions for specific points in time. Therefore, 
             I created this website to play with the ARIMA model.
            """)

    st.subheader("The purpose")
    st.write("""\n
        The goal of this app is to allow an user to experiment easily with ARIMA models. Thanks to that, the app helps
         to understand: \n
        - when it is efficient to use the model,
        - which scenarios are more straightforward for the model to predict, and which are more complex,
        - how the noise interrupts the predictions,
        - how important is the right balance between autoregression, differencing, and moving average parts of
          the model,
        - how we can set up the pieces from the previous point,
        - understanding that the increasing hyperparameters' value in the model configuration only sometimes leads to
          better performance.
        """)

    st.subheader("Typical use case")
    st.write("""
        - Create a synthetic dataset using various choices to define the characteristics.
        - See and validate the dataset to understand the data.
        - Set up model hyperparameters.
        - Train the model and make predictions.
        - Compare the predicted values with the actual values.
        \n""")

    st.subheader(""" \n Use the radio buttons on the left to navigate between chapters. \n \n \n \n""")

elif choice == 1:
    # To be finished up later
    st.title(""" ARIMA model""")
    st.subheader("The goal - time-series forecasting")
    st.write("""\n
        Imagine a scenario where production company X wants to optimize its production and storage costs. So the
         company wants to know their future sales. The company has recorded their main product sales up to the current
          week - which you can see as the solid line on the graph below. \n
        Your task (as a machine learning specialist) is to predict future sales based on the data available. You can
         easily see that the general sales are going up but also there is some seasonality in the data. Knowing that
         you decided to search for a model which can consider these factors. But also you don't want to
         use a complex model as the dataset is small. Also company X wants to understand how the prediction
         is done.  
        """)
    st.image('timeseries.png', width=600, caption="Solid line - known data, dotted - data you want to predict "
                                                  "(but you don't know it yet)")
    st.subheader("How is it built?")
    st.write("""\n
            After thoughtful research, you decided that the ARIMA model is a good choice for the case. The ARIMA model 
            is divided into three parts which are later added together:
             - **AR** - **A**uto-**R**egressive - simple linear regression using past values. **Hyperparameter p** 
               decides how many previous values are taken into account to predict the next value. 
             - **I** - **I**ntegrated - to make the model suitable for non-stationary tasks (like our case) we need to
               stationarize the predictions. **Hyperparameter d** decides which differencing order is used. For
               example: using first-order differencing makes the model predict how much the next value is different
               from the previous one instead of directly predicting the next value.
             - **MA** - **M**oving **A**verage - learns how much the model was off the target in the previous steps.
               Then provides the adjustment to account for the error in the future predictions. **Hyperparameter q** 
               defines how many previous predictions are taken into account while calculating the correction.
            \n
            So we use three variables to define a model - ARIMA(p, d, q). So for example ARIMA(2, 1, 0) means:
            - 2: Two previous points are used for making a regression
            - 1: First-order differencing is used to predict values
            - 1: The average moving window takes only the last error to create a correction value.
            \n
            """)
    st.write('In this case, we can write the equation:')
    st.latex("""Y_t - Y_{t-1} = \phi_1(Y_{t - 1} - Y_{t - 2}) + \phi_2(Y_{t - 2} - Y_{t - 3}) 
                + θ_{1}\epsilon_{t - 1} + \epsilon_t""")
    st.write(""" Where:
                 \n- Y - time-series value
                 \n - ϕ - autoregressive coefficient (AR)
                 \n- ϵ - error term (MA)
                 \n- θ - moving average coefficient (MA)
                 \n - (I) is represented by predicting the difference istead of exact value
            """)

else:
    # Print instructions
    data_setup, model_setup = populate_sidebar_get_setup()
    points_granularity = data_setup['points_granularity']
    st.subheader("Instructions")
    st.write("""
            1. Modify the dataset using sliders in the dataset group in the sidebar.
            2. Use the "Validate dataset" button to check if the dataset is stationary. Scroll the sidebar down
               to see the buttons.
            3. Based on the findings, set the model parameters and press the "Train and predict" button. You will see
               the predictions below the original graph.
            4. If the model predictions are off, modify the model. If the predictions are accurate, make the dataset
               more challenging.
            5. Good luck, and have fun!
            \n""")

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
