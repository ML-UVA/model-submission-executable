from load_dataloader import *
from model_eval import Eval
from firebase import Firebase

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from argparse import ArgumentParser

parser = ArgumentParser(description='Process model submission information')

parser.add_argument('name', type=str, help='Your name')
parser.add_argument('computing_id', type=str, help='Your computing id (just the digits, no @virginia.edu)')
parser.add_argument('path', type=str, help='Path to your trained model (ex: path/to/model.pt)')
parser.add_argument('competition', type=int, help='Competition ID whose dataset your model has been trained on')

args = parser.parse_args()

metrics = {
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'r2': r2_score
}
f = Firebase()
competition = f.get_competition(args.competition)
score_func = competition['function']

model_load = Eval(args.path, competition)
res = model_load.eval([(name, metrics[name]) for name in score_func])

print('Your results:')
for name in res:
    print(f'{name}: {res[name]:.4f}')

f.add_submission({
    'name': args.name,
    'computing_id': args.computing_id,
    'competition': args.competition,
    'score': sum(score_func[metric] * res[metric] for metric in score_func),
    'metrics': res
})
print('Done')
