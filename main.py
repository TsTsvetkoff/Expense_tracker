from flask import Flask, render_template, request, redirect, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
from sqlalchemy import func
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///expenses.db'
db = SQLAlchemy(app)


class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    category = db.Column(db.String(50), nullable=False)
    comment = db.Column(db.String(200), nullable=True)
    amount = db.Column(db.Float, nullable=False)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        category = request.form.get('category')
        comment = request.form.get('comment')
        amount = request.form.get('amount')
        new_expense = Expense(category=category, comment=comment, amount=amount)
        db.session.add(new_expense)
        db.session.commit()
        return redirect('/')
    expenses = Expense.query.all()
    predicted_value = predict_next_day_expense()
    return render_template('index.html', expenses=expenses, predicted_value=predicted_value)


def predict_next_day_expense():
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

        return round(predicted_amount[0][0], 2)


@app.route('/data')
def data():
    from sqlalchemy import func
    categories = db.session.query(Expense.category, func.sum(Expense.amount)).group_by(Expense.category).all()
    labels = [category[0] for category in categories]
    amounts = [category[1] for category in categories]
    return {'labels': labels, 'amounts': amounts}


@app.route('/expenses')
def expenses():
    expenses = Expense.query.order_by(Expense.date.desc()).all()
    return jsonify([{
        'date': expense.date.strftime('%Y-%m-%d'),
        'category': expense.category,
        'comment': expense.comment,
        'amount': expense.amount
    } for expense in expenses])


@app.route('/total_expenses')
def total_expenses():
    today = datetime.utcnow().date()
    total = db.session.query(func.sum(Expense.amount)).filter(func.date(Expense.date) == today).scalar()
    return jsonify({'total': total or 0})


@app.route('/expenses_per_day')
def expenses_per_day():
    expenses_per_day = db.session.query(
        func.date(Expense.date).label('date'),  # Group by date
        func.sum(Expense.amount).label('total'),
        func.coalesce(func.sum(Expense.amount).filter(Expense.category == 'Кафе / Ресторант'), 0).label('cafe'),
        func.coalesce(func.sum(Expense.amount).filter(Expense.category == 'Магазин / Табаче'), 0).label('shop'),
        func.coalesce(func.sum(Expense.amount).filter(Expense.category == 'Деца'), 0).label('kids'),
        func.coalesce(func.sum(Expense.amount).filter(Expense.category == 'Сметки'), 0).label('bills'),
        func.coalesce(func.sum(Expense.amount).filter(Expense.category == 'Други'), 0).label('others')
    ).group_by(func.date(Expense.date)).order_by(func.date(Expense.date).desc()).all()

    if not expenses_per_day:
        dates = []
        totals = []
    else:
        dates = [expense.date for expense in expenses_per_day]
        totals = [expense.total if expense.total else 0 for expense in expenses_per_day]

    predicted_value = predict_next_day_expense()

    return render_template('expenses_per_day.html', expenses_per_day=expenses_per_day, dates=dates, totals=totals, predicted_value=predicted_value)


@app.route('/expenses_per_month')
def expenses_per_month():
    expenses_per_month = db.session.query(
        func.strftime("%Y-%m", Expense.date).label('month'),  # Group by month and year
        func.sum(Expense.amount).label('total'),
        func.coalesce(func.sum(Expense.amount).filter(Expense.category == 'Кафе / Ресторант'), 0).label('cafe'),
        func.coalesce(func.sum(Expense.amount).filter(Expense.category == 'Магазин / Табаче'), 0).label('shop'),
        func.coalesce(func.sum(Expense.amount).filter(Expense.category == 'Деца'), 0).label('kids'),
        func.coalesce(func.sum(Expense.amount).filter(Expense.category == 'Сметки'), 0).label('bills'),
        func.coalesce(func.sum(Expense.amount).filter(Expense.category == 'Други'), 0).label('others')
    ).group_by(func.strftime("%Y-%m", Expense.date)).order_by(func.strftime("%Y-%m", Expense.date).desc()).all()

    if not expenses_per_month:
        months = []
        totals = []
    else:
        months = [expense.month for expense in expenses_per_month]
        totals = [expense.total if expense.total else 0 for expense in expenses_per_month]

    return render_template('expenses_per_month.html', expenses_per_month=expenses_per_month, months=months, totals=totals)



if __name__ == "__main__":
    with app.app_context():
        if not os.path.exists('expenses.db'):
            db.create_all()
    app.run(debug=True)