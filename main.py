from flask import Flask, render_template, request, redirect, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
from sqlalchemy import func

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
    return render_template('index.html', expenses=expenses)


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

    return render_template('expenses_per_day.html', expenses_per_day=expenses_per_day, dates=dates, totals=totals)




@app.route('/expenses_per_month')
def expenses_per_month():
    expenses_per_month = db.session.query(Expense.category, func.strftime('%Y-%m', Expense.date), func.sum(Expense.amount)).group_by(Expense.category, func.strftime('%Y-%m', Expense.date)).all()
    return render_template('expenses_per_month.html', expenses_per_month=expenses_per_month)


if __name__ == "__main__":
    with app.app_context():
        if not os.path.exists('expenses.db'):
            db.create_all()
    app.run(debug=True)