from flask import Flask, render_template, request, redirect, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
from sqlalchemy import func
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sqlalchemy import text

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///expenses.db'
db = SQLAlchemy(app)


with app.app_context():
    try:
        db.session.execute(
            text('ALTER TABLE expense ADD COLUMN prediction FLOAT DEFAULT NULL')
        )
        db.session.commit()
    except Exception as e:
        print(f"An error occurred: {e}")


class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    category = db.Column(db.String(50), nullable=False)
    comment = db.Column(db.String(200), nullable=True)
    amount = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.Float)


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

        date_to_update = df['date'].max().strftime('%Y-%m-%d %H:%M:%S')

        predicted_amount = round(predicted_amount[0][0], 2)
        db.session.execute(
            text(
                'UPDATE expense SET prediction = :predicted_amount WHERE strftime("%Y-%m-%d %H:%M:%S", date) = :date_to_update'),
            {'predicted_amount': predicted_amount, 'date_to_update': date_to_update}
        )
        db.session.commit()

        return round(predicted_amount)


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
        predictions = []
    else:
        dates = [expense.date for expense in expenses_per_day]
        totals = [expense.total if expense.total else 0 for expense in expenses_per_day]
        predictions = [get_last_prediction_of_day(date) for date in dates]  # Get predictions

    predicted_value = predict_next_day_expense()

    return render_template(
        'expenses_per_day.html',
        expenses_per_day=expenses_per_day,
        dates=dates,
        totals=totals,
        predictions=predictions,  # Pass predictions as a list
        predicted_value=predicted_value
    )


def get_last_prediction_of_day(day):
    # Convert the string date to a datetime object
    day_datetime = datetime.strptime(day, '%Y-%m-%d')

    # Convert the date to a datetime object at the start of the day
    start_of_day = datetime.combine(day_datetime.date(), datetime.min.time())

    # Convert the date to a datetime object at the end of the day
    end_of_day = datetime.combine(day_datetime.date(), datetime.max.time())

    # Query the database for expenses on the given day, ordered by date and time (from newest to oldest)
    expenses_of_day = db.session.query(Expense).filter(
        Expense.date >= start_of_day,
        Expense.date <= end_of_day
    ).order_by(Expense.date.desc())

    # Get the first entry (i.e., the last prediction of the day)
    last_expense_of_day = expenses_of_day.first()

    # Return the prediction value of the last expense of the day
    return last_expense_of_day.prediction if last_expense_of_day else None


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
