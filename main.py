from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sqlalchemy import text
import numpy as np
#from transformers import pipeline
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///expenses.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    comment = db.Column(db.String(200))

class DailySummary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    restaurant_coffee = db.Column(db.Float, default=0)
    shop = db.Column(db.Float, default=0)
    kids = db.Column(db.Float, default=0)
    bills = db.Column(db.Float, default=0)
    other = db.Column(db.Float, default=0)
    prediction = db.Column(db.Float, default=0)
    ai_feedback = db.Column(db.Text)

class MonthlySummary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer, nullable=False)
    month = db.Column(db.Integer, nullable=False)
    restaurant_coffee = db.Column(db.Float, default=0)
    shop = db.Column(db.Float, default=0)
    kids = db.Column(db.Float, default=0)
    bills = db.Column(db.Float, default=0)
    other = db.Column(db.Float, default=0)

def predict_next_day_expense():
    expenses = Expense.query.with_entities(Expense.date, Expense.amount).all()
    df = pd.DataFrame(expenses, columns=['date', 'amount'])

    # Ensure date column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    min_date = df['date'].min()
    df_daily_totals = df.groupby(df['date'].dt.date)['amount'].sum().reset_index()
    df_daily_totals.columns = ['date', 'total_amount']

    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_daily_totals['date']):
        df_daily_totals['date'] = pd.to_datetime(df_daily_totals['date'])

    # Convert the 'date' column to the number of days since the minimum date
    min_date_value = df_daily_totals['date'].min()
    df_daily_totals['date'] = (df_daily_totals['date'] - min_date_value).dt.days

    # Split the data into input and output
    X = df_daily_totals['date'].values.reshape(-1, 1)
    y = df_daily_totals['total_amount'].values.reshape(-1, 1)

    # Split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Set current_date to the last date in df_daily_totals
    current_date = df_daily_totals['date'].max()

    # Predict for the next day
    next_day = current_date + 1

    # Reshape next_day
    next_day = np.array(next_day).reshape(-1, 1)
    predicted_amount = model.predict(next_day)

    predicted_amount = round(predicted_amount[0][0], 2)

    # Update the DailySummary table instead of the Expense table
    latest_date = df['date'].max().date()
    daily_summary = DailySummary.query.filter_by(date=latest_date).first()
    if daily_summary:
        daily_summary.prediction = predicted_amount
        db.session.commit()

    return round(predicted_amount)


def get_ai_feedback(expenses, total_today):
    # TODO: Implement AI feedback based on spending patterns
    return None

# def get_ai_feedback(expenses, total_today):
#     if not expenses:
#         return None
#
#     # Initialize sentiment analysis pipeline
#     classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')
#
#     # Prepare spending analysis text
#     spending_pattern = f"Today's total spending is ${total_today:.2f} compared to daily target of 100.00. "
#     if total_today > 100:
#         spending_pattern += f"This is {total_today - 100:.2f} over the daily target."
#     else:
#         spending_pattern += f"This is {100 - total_today:.2f} under the daily target."
#
#     # Get sentiment and generate feedback
#     result = classifier(spending_pattern)[0]
#     sentiment = result['label']
#
#     if sentiment == 'POSITIVE':
#         feedback = "Great job managing your expenses today! You're staying within your budget."
#     else:
#         feedback = "Consider reviewing your spending habits to stay within your daily target."
#
#     return feedback

@app.route('/')
@app.route('/expenses')
def expenses():
    today = datetime.now().date()
    current_month = today.replace(day=1)

    # Get all expenses
    expenses = Expense.query.order_by(Expense.date.desc()).all()

    categories = ['Restaurant / Coffee', 'Shop', 'Kids', 'Bills', 'Other']

    # Get or create today's daily summary
    daily_summary = DailySummary.query.filter_by(date=today).first()
    if not daily_summary:
        daily_summary = DailySummary(
            date=today,
            restaurant_coffee=0.0,
            shop=0.0,
            kids=0.0,
            bills=0.0,
            other=0.0,
            prediction=0.0
        )
        db.session.add(daily_summary)

    # Always recalculate prediction for today
    recent_daily_summaries = DailySummary.query.filter(DailySummary.date <= today).order_by(DailySummary.date.desc()).all()
    daily_amounts = [sum([s.restaurant_coffee, s.shop, s.kids, s.bills, s.other]) for s in recent_daily_summaries]
    daily_prediction = predict_next_day_expense()
    if daily_prediction is not None:
        daily_summary.prediction = daily_prediction
        db.session.commit()

    # Calculate total expenses for today
    today_expenses = Expense.query.filter_by(date=today).all()
    total_expenses = sum(expense.amount for expense in today_expenses) if today_expenses else 0

    return render_template('expenses.html',
                           expenses=expenses,
                           categories=categories,
                           daily_prediction=daily_summary.prediction if daily_summary else None,
                           total_expenses=total_expenses,
                           ai_feedback=daily_summary.ai_feedback if daily_summary else None)

@app.route('/add_expense', methods=['POST'])
def add_expense():
    date_str = request.form['date']
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        # If the date string includes time, parse it differently
        date = datetime.strptime(date_str.split('.')[0], '%Y-%m-%d %H:%M:%S').date()
    category = request.form['category']
    amount = float(request.form['amount'])
    comment = request.form.get('comment')

    expense = Expense(date=date, category=category, amount=amount, comment=comment)
    db.session.add(expense)

    # Update daily summary
    daily_summary = DailySummary.query.filter_by(date=date).first()
    if not daily_summary:
        daily_summary = DailySummary(
            date=date,
            restaurant_coffee=0.0,
            shop=0.0,
            kids=0.0,
            bills=0.0,
            other=0.0,
            prediction=0.0
        )
        db.session.add(daily_summary)

    # Get recent daily summaries for predictions
    recent_daily_summaries = DailySummary.query.filter(DailySummary.date <= date).order_by(DailySummary.date.desc()).all()
    daily_amounts = [sum([s.restaurant_coffee, s.shop, s.kids, s.bills, s.other]) for s in recent_daily_summaries]

    # Calculate prediction for this date
    daily_prediction = predict_expenses(daily_amounts) if daily_amounts else None
    if daily_prediction is not None:
        daily_summary.prediction = daily_prediction

    # Get AI feedback on spending patterns
    today_total = sum(exp.amount for exp in Expense.query.filter_by(date=date).all()) + amount
    #ai_feedback = get_ai_feedback(recent_expenses, today_total)
    #daily_summary.ai_feedback = ai_feedback

    # Update category amounts
    if category == 'Restaurant / Coffee':
        daily_summary.restaurant_coffee += amount
    elif category == 'Shop':
        daily_summary.shop += amount
    elif category == 'Kids':
        daily_summary.kids += amount
    elif category == 'Bills':
        daily_summary.bills += amount
    else:
        daily_summary.other += amount

    # Update monthly summary
    monthly_summary = MonthlySummary.query.filter_by(
        year=date.year,
        month=date.month
    ).first()

    if not monthly_summary:
        monthly_summary = MonthlySummary(
            year=date.year,
            month=date.month,
            restaurant_coffee=0.0,
            shop=0.0,
            kids=0.0,
            bills=0.0,
            other=0.0
        )
        db.session.add(monthly_summary)

    # Update monthly summary amounts
    if category == 'Restaurant / Coffee':
        monthly_summary.restaurant_coffee += amount
    elif category == 'Shop':
        monthly_summary.shop += amount
    elif category == 'Kids':
        monthly_summary.kids += amount
    elif category == 'Bills':
        monthly_summary.bills += amount
    else:
        monthly_summary.other += amount

    db.session.commit()
    return redirect(url_for('expenses'))

@app.route('/daily_summary')
def daily_summary():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    pagination = DailySummary.query.order_by(DailySummary.date.desc()).paginate(page=page, per_page=per_page)
    summaries = pagination.items
    return render_template('daily_summary.html',
                         summaries=summaries,
                         pagination=pagination,
                         per_page=per_page)

@app.route('/monthly_summary')
def monthly_summary():
    summaries = MonthlySummary.query.order_by(
        MonthlySummary.year.desc(),
        MonthlySummary.month.desc()
    ).all()

    # Prepare data for monthly prediction
    monthly_prediction = None
    if summaries and len(summaries) >= 2:  # Ensure we have at least 2 data points
        monthly_totals = []
        for s in summaries:
            if all(v is not None for v in [s.restaurant_coffee, s.shop, s.kids, s.bills, s.other]):
                total = sum([s.restaurant_coffee, s.shop, s.kids, s.bills, s.other])
                monthly_totals.append(total)

        if len(monthly_totals) >= 2:  # Double check we still have enough valid data points
            monthly_prediction = predict_expenses(monthly_totals, 'monthly')

    return render_template('monthly_summary.html',
                           summaries=summaries,
                           monthly_prediction=monthly_prediction)

def predict_expenses(amounts, period='daily'):
    """
    Predict future expenses based on historical data.

    Args:
        amounts: List of historical expense amounts
        period: 'daily' or 'monthly' to indicate prediction period

    Returns:
        Predicted amount for the next period
    """
    if not amounts or len(amounts) < 2:
        return None

    # Convert to numpy array and reshape for sklearn
    X = np.array(range(len(amounts))).reshape(-1, 1)
    y = np.array(amounts).reshape(-1, 1)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict next value
    next_period = np.array([[len(amounts)]])
    predicted_amount = model.predict(next_period)

    return round(float(predicted_amount[0][0]), 2)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
