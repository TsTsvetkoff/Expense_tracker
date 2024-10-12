from main import app, Expense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np


with app.app_context():
    expenses = Expense.query.with_entities(Expense.date, Expense.amount).all()

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(expenses, columns=['date', 'amount'])

    min_date = df['date'].min()
    # Convert the 'date' column back to datetime
    df['date'] = pd.to_datetime(df['date'], unit='D')

    # Calculate the timedelta
    min_date_timedelta = pd.to_timedelta(min_date.value)

    # Add min_date to each element in the 'date' column
    df['date'] = df['date'].apply(lambda x: x + min_date_timedelta)

    # Group by 'date' and sum 'amount'
    df_daily_totals = df.groupby(df['date'].dt.date)['amount'].sum().reset_index()

    # Rename the columns for clarity
    df_daily_totals.columns = ['date', 'total_amount']

    # Convert the 'date' column to the number of days since the minimum date
    df_daily_totals['date'] = (df_daily_totals['date'] - df_daily_totals['date'].min()) / np.timedelta64(1, 'D')

    # Split the data into input and output
    X = df_daily_totals['date'].values.reshape(-1, 1)
    y = df_daily_totals['total_amount'].values.reshape(-1, 1)

    # Split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)


    # Convert the 'date' column to the number of days since the minimum date
    df_daily_totals['date'] = (df_daily_totals['date'] - df_daily_totals['date'].min())

    # Set current_date to the last date in df_daily_totals
    current_date = df_daily_totals['date'].max()

    # Predict for the next day
    next_day = current_date + 1

    # Reshape next_day
    next_day = np.array(next_day).reshape(-1, 1)
    predicted_amount = model.predict(next_day)

    print(f"The predicted total expense for the next day is: {predicted_amount}")