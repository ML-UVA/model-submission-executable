from creds import get_key, get_db_name

from firebase_admin import initialize_app
from firebase_admin import credentials
from firebase_admin import db
import time

class Firebase:
    def __init__(self):
        cred = credentials.Certificate(get_key())

        initialize_app(cred, {
            'databaseURL': get_db_name()
        })
        self.ref = db.reference()

    def add_submission(self, data):
        submission_ref = self.ref.child('submissions')
        data['time'] = time.time()
        existing = submission_ref.order_by_child('computing_id').equal_to(data['computing_id']).limit_to_first(1).get()
        if existing == None or len(existing) == 0:
            new_submission = submission_ref.push()
        else:
            for key in existing:
                new_submission = submission_ref.child(key)
            if new_submission.get()['score'] >= data['score']:
                print('Performed worse than previous submission - not replacing it')
                return
        new_submission.set(data)
        print('Results successfully uploaded')

    def get_score_function(self, id):
        score_ref = self.ref.child('score_functions')
        data = score_ref.order_by_child('competition_id').equal_to(id).limit_to_first(1).get()
        if data == None or len(data) == 0:
            new_func = score_ref.push()
            new_func.set({
                'competition_id': id,
                'function': {
                    'mse': 0.33,
                    'mae': 0.33,
                    'r2': 0.33
                }
            })
            return {
                'mse': 0.33,
                'mae': 0.33,
                'r2': 0.33
            }
        else:
            for key in data:
                return data[key]['function']
            