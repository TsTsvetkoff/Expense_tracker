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


if __name__ == "__main__":
    with app.app_context():
        if not os.path.exists('expenses.db'):
            db.create_all()
    app.run(debug=True)