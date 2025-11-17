
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///people.db'
db = SQLAlchemy(app)

class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    dob = db.Column(db.Date, nullable=True)
    folder = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f'<Person {self.name}>'


def add_person(name, email, phone, dob, folder):
    with app.app_context():
        from datetime import datetime
        # Parse DOB string to date object if provided
        dob_date = None
        if dob:
            try:
                dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
            except ValueError:
                print(f"Invalid date format for {name}: {dob}")
        
        person = Person(name=name, email=email, phone=phone, dob=dob_date, folder=folder)
        db.session.add(person)
        db.session.commit()
        print(f'Added {name} to the database.')

if __name__ == '__main__':
    import sys
    with app.app_context():
        db.create_all()
        if len(sys.argv) == 6:
            # Usage: python models.py Name Email Phone DOB FolderName
            name, email, phone, dob, folder = sys.argv[1:6]
            add_person(name, email, phone, dob, folder)
        else:
            print('Database and tables created.')
            print('To add a person: python models.py Name Email Phone DOB FolderName')
