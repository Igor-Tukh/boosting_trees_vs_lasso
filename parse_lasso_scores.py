import os
import argparse

def parse_scores(filepath):
    results = {}
    with open(filepath, 'r') as scores_file:
        lines = [line.replace(os.linesep, '') for line in scores_file.readlines()]
        break_ind = None
        for i, line in enumerate(lines):
            if line == '==================================================':
                break_ind = i
        lines = lines[break_ind + 2:]
        for line in lines:
            tokens = line.replace(' ', '\t').split('\t')
            if tokens[-1] == '':
                tokens = tokens[:-1]
            if len(tokens) == 5:
                subset = f'{tokens[1]}_rmse'
                lambda_coef = float(tokens[2])
                iter_number = int(tokens[0])
                score = float(tokens[4].replace(':', "=").split("=")[1])
                if lambda_coef not in results:
                    results[(lambda_coef, iter_number)] = {}
                results[(lambda_coef, iter_number)][subset] = score
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores_path', type=str, required=True)
    parser.add_argument('--get_best_lambda', action='store_true', required=False)
    parser.add_argument('--with_lambda', type=str, required=False)
    parser.add_argument('--with_iter_number', type=str, required=False)
    args = parser.parse_args()
    scores = parse_scores(args.scores_path)
    best_lambda = None    
    best_score = None
    best_iter_number = None
    eps = 1e-9
    if args.get_best_lambda:
        for (lambda_coef, iter_number), scores in scores.items():
            if best_score is None or best_score > scores['test_rmse']:
                best_score = scores['test_rmse']
                best_lambda = lambda_coef
                best_iter_number = iter_number
        print(best_lambda, iter_number)
    if args.with_lambda and args.with_iter_number:
        target_lambda = float(args.with_lambda)
        targer_iter_number = int(args.with_iter_number)
        closest_lambda = None
        closest_iter_number = None
        for lambda_coef, iter_number in scores.keys():
            if (closest_lambda is None or abs(closest_lambda - target_lambda) > abs(lambda_coef - target_lambda)):
                closest_lambda = lambda_coef
                closest_iter_number = iter_number
            if abs(lambda_coef - target_lambda) < abs(closest_lambda - target_lambda) + eps and (abs(closest_iter_number - targer_iter_number) > abs(iter_number - targer_iter_number)):
                closest_iter_number = iter_number
                closest_lambda = lambda_coef
        print(scores[(closest_lambda, closest_iter_number)]['test_rmse'])
        
