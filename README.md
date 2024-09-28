# Expense Tracker

This project is a simple expense tracker built with Python and Flask. 
It allows users to input their expenses and categorize them. The expenses are stored in a SQLite database and can be viewed in a table format on the web page.
The project also includes a pie chart visualization of expenses by category.

## Technologies Used

- Python
- Flask
- SQLAlchemy
- HTML/CSS/JavaScript

## Features

- Add new expenses with category, comment, and amount.
- View all expenses in a table format.
- Visualize expenses by category in a pie chart.
- Calculate and display total expenses for the current day.

## Setup

1. Clone the repository to your local machine.
2. Install the required Python packages using pip:

```bash
pip install flask flask_sqlalchemy
```

3. Run the application:

```bash
python main.py
```

4. Open your web browser and navigate to `http://localhost:5000` to view the application.

## File Structure

- `main.py`: This is the main Python file that runs the Flask application. It includes the routes for the application and the SQLAlchemy model for the expenses.
- `templates/index.html`: This is the HTML template for the main page of the application. It includes the form for adding new expenses and the table for displaying all expenses.
- `expenses.db`: This is the SQLite database file where all the expenses are stored.

## Future Improvements

- Add user authentication to allow multiple users to track their expenses separately.
- Add more detailed reporting features, such as viewing expenses by month or year.
- Improve the UI/UX design of the application.
